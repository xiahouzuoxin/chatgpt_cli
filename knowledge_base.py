import os
import re
import warnings
import concurrent
import multiprocessing
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import mimetypes
import dill

import docx2txt
from PyPDF2 import PdfReader

from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

cpu_count = multiprocessing.cpu_count()

class Embedding:
    def __init__(self, source='sentence_transformer', model='all-MiniLM-L6-v2') -> None:
        self.param_source = source
        self.param_model = model
        if source == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
            self._encode = lambda text: list(self.model.encode(text))
        elif source == 'openai':
            # "text-embedding-ada-002"
            self.model = model
            self._encode = self._get_gpt_embedding

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_gpt_embedding(self, text: str) -> list[float]:
        return openai.Embedding.create(input=[text], model=self.model)["data"][0]["embedding"]
    
    def encode(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Embedding input text must be a non-empty string: {text}")
        result = self._encode(text)
        return result

    def get_embedding(self, text: str):
        return self.encode(text)
    
    def get_embedding_parallel(self, texts: str, num_workers=cpu_count):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda text: self.encode(text), 
                        texts
                    ), 
                    total=len(texts)
                )
            )
        return results
    
    def cal_gpt_embedding_price(df, text_col='text', base_price_1k=0.0004):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        df['n_tokens'] = df[text_col].apply(lambda x: len(tokenizer.encode(x)))
        total_tokens = df['n_tokens'].sum()
        return base_price_1k * (total_tokens / 1000), total_tokens
    
class EmbeddingDoc(BaseDoc):
    doc: str = None
    text: str
    embedding: NdArray[384] # embedding dim of emb_model: sentence_transformer/all-MiniLM-L6-v2

class VectorRetrieval():
    def __init__(self, workspace='./knowledge_db/') -> None:
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        
        # Specify your workspace path
        self.db = InMemoryExactNNVectorDB[EmbeddingDoc](workspace=workspace)

        # embedding model
        # Reference: https://www.sbert.net/docs/pretrained_models.html
        self.emb_model_config = {
            'source': 'sentence_transformer',
            'model': 'all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'max_seq_len': 256
        }
        self.emb_model = Embedding(source=self.emb_model_config['source'], model=self.emb_model_config['model'])

    def add_index_for_texts(self, texts: list, num_workers=cpu_count):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            doc_list = list(
                tqdm(
                    executor.map(
                        lambda _text: EmbeddingDoc(
                            text=_text, 
                            embedding=self.emb_model.get_embedding(_text)
                        ), 
                        texts
                    ), 
                    total=len(texts)
                )
            )
        # doc_list = [EmbeddingDoc(
        #         text=_text, 
        #         embedding=self.emb_model.get_embedding(_text)
        #     ) for _text in texts]

        self.db.index(inputs=DocList[EmbeddingDoc](doc_list))

    def add_index_for_docs(self, files: list, num_workers=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda file: self.extract_chunk_texts_from_file(file), 
                        files
                    ), 
                    total=len(files)
                )
            )
        self.add_index_for_texts([text for texts in results for text in texts])

    # Extract text from a file based on its mimetype
    def extract_chunk_texts_from_file(self, file):
        """Return the text content of a file."""
        file_mimetype = mimetypes.guess_type(file)[0]
        if file_mimetype == "application/pdf":
            # Extract text from pdf using PyPDF2
            reader = PdfReader(file)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text()
        elif file_mimetype == "text/plain":
            # Read text from plain text file
            extracted_text = file.read().decode("utf-8")
            file.close()
        elif file_mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Extract text from docx using docx2txt
            extracted_text = docx2txt.process(file)
        else:
            # Unsupported file type
            extracted_text = None
            warnings.warn("Unsupported file type of {}: {}".format(file, file_mimetype))
        if extracted_text is None:
            return []

        text_chunks = self.chunks(extracted_text, self.emb_model_config['max_seq_len'])
        return text_chunks
    
    def chunks(self, text, max_seq_len, overlap=0.5):
        texts = re.split('\.|\n', text.replace("  ", " "))
        text_chunks = []
        for _text in texts:
            if len(_text) > max_seq_len:
                k = 0
                while k < len(_text):
                    # TODO: find , or space to chunk
                    text_chunks.append(_text[k:min(k+max_seq_len, len(_text))])
                    k += int(max_seq_len * overlap)
            elif len(_text.strip(',').strip(' ')) > 5: # ignore short text
                text_chunks.append(_text)
        return text_chunks

    def query_local(self, text, limit=10):
        # Perform a search query
        query = EmbeddingDoc(
                text=text, 
                embedding=self.emb_model.get_embedding(text)
            )
        results = self.db.search(inputs=DocList[EmbeddingDoc]([query]), limit=limit)
        return [_rt.matches for _rt in results]

    def serve(self):
        # Serve the DB
        # How to search from Client?
        # ```
        #   from vectordb import Client
        #   # Instantiate a client connected to the server. In practice, replace 0.0.0.0 to the server IP address.
        #   client = Client[ToyDoc](address='grpc://0.0.0.0:12345')
        #   # Perform a search query
        #   results = client.search(inputs=DocList[ToyDoc]([query]), limit=10)
        # ```

        with self.db.serve(protocol='grpc', port=12345, replicas=1, shards=1) as service:
            service.block()

    def query_served(self, text, limit=10, address='grpc://0.0.0.0:12345'):
        from vectordb import Client
        client = Client[EmbeddingDoc](address=address)
        # Perform a search query
        query = EmbeddingDoc(
                text=text, 
                embedding=self.emb_model.get_embedding(text)
            )
        results = client.search(inputs=DocList[EmbeddingDoc]([query]), limit=limit)
        return [_rt.matches for _rt in results]

def get_knowledge_base(source_dir='./knowledge/docs'):
    vector_retrieval = VectorRetrieval('./knowledge/vector_db/')

    docs = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            docs.append( os.path.join(root, file) )
    vector_retrieval.add_index_for_docs(docs)
    print('Generate knowledge base down.')
    
    return vector_retrieval

if __name__ == '__main__':
    get_knowledge_base()