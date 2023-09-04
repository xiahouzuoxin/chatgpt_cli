import os
import re
import warnings
import hashlib
import concurrent
import multiprocessing
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import mimetypes
import chromadb
import docx2txt
from PyPDF2 import PdfReader

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
        if self.param_source == 'sentence_transformer':
            # convert np.float to float
            result = [v.item() for v in result]
        return result

    def get_embedding(self, texts: list):
        return [self.encode(text) for text in texts]
    
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

class VectorRetrieval():
    def __init__(self, workspace) -> None:
        if not os.path.exists(workspace):
            os.mkdir(workspace)

        self.chroma_client = chromadb.PersistentClient(path=workspace)
        self.collection = self.chroma_client.get_or_create_collection(name='knowledge_base')

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
        texts = list(set(texts))
        num_workers = min(num_workers, len(texts))
        text_embs = self.emb_model.get_embedding_parallel(texts, num_workers=num_workers)
        ids = [hashlib.sha1(_text.encode("utf-8")).hexdigest() for _text in texts]
        self.collection.add(documents=texts, embeddings=text_embs, ids=ids)

    def add_index_for_docs(self, path: str, num_workers=5):
        docs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                docs.append( os.path.join(root, file) )
        if len(docs) == 0:
            return 0
        num_workers = min(num_workers, len(docs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda doc: self.extract_chunk_texts_from_file(doc), 
                        docs
                    ), 
                    total=len(docs)
                )
            )
        texts = [_text for _texts in results for _text in _texts]
        self.add_index_for_texts()
        return len(texts)

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
            _text = _text.strip(' ').strip(',')
            if len(_text) <= 5:
                continue
            if len(_text) > max_seq_len:
                k = 0
                while k < len(_text):
                    # TODO: find , or space to chunk
                    text_chunks.append(_text[k:min(k+max_seq_len, len(_text))])
                    k += int(max_seq_len * overlap)
            else:
                text_chunks.append(_text)
        return text_chunks
    
    def query(self, texts: str, limit=3):
        query_embs = self.emb_model.get_embedding_parallel(texts, num_workers=min(len(texts), cpu_count))
        return self.collection.query(query_embeddings=query_embs, n_results=limit)

if __name__ == '__main__':
    vector_retrieval = VectorRetrieval('./knowledge/vector_db/')
    vector_retrieval.add_index_for_docs(path='./knowledge/docs')
    print('Generate knowledge base down.')
    
    print( vector_retrieval.query(['How are you'], limit=1) )