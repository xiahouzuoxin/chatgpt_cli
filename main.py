import argparse
from datetime import datetime
import openai

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', help='openai model, default gpt-3.5-turbo')
parser.add_argument('--max_tokens', required=False, type=int, default=1024, help='max_tokens, default 1024')
parser.add_argument('--temperature', required=False, type=float, default=0, help='temperature, default 0')
parser.add_argument('--max_history_len', required=False, type=int, default=5, help='max history length, default 5')
args = parser.parse_args()

current_date = datetime.now().strftime('%Y-%m-%d')
system_prompt = f'''
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: {current_date}
'''

import abc

class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f">> {role}: ")

    def prompt_for_output(self, role: str):
        print(f">> {role}: ", end="", flush=True)

    def batch_output(self, output_text):
        print(output_text, flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

messaegs = [
    {"role": "system", "content": system_prompt},
]

chatio = SimpleChatIO()
while True:
    chatio.prompt_for_output(messaegs[-1]["role"])
    chatio.batch_output(messaegs[-1]["content"])
    try:
        inp = chatio.prompt_for_input("user")
    except EOFError:
        inp = ""
    if not inp:
        print("exit...")
        break

    messaegs.append({"role": "user", "content": inp})
    response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=messaegs[-args.max_history_len:],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
    answer = response['choices'][0]['message']['content']
    messaegs.append({"role": "system", "content": answer})
