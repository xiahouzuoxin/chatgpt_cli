from datetime import datetime

SYSTEM_PROMPT = f'''
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: {datetime.now().strftime('%Y-%m-%d')}
'''

KNOWLEDGE_PROMPT = f'''You can use the following knowledge to chat with me:
<<KNOWLEDGE>>
'''