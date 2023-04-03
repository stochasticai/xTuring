import openai
import os
from typing import List
import ast
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion_from_chatgpt(
    messages: List,
    openai_key: str = None
):  
    if openai_key is not None:
        openai.api_key = openai_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    
    return response["choices"][0]["message"]['content']


def instruction_input_suggest(
        original_text: str,
        openai_key: str = None,
        chunk_size: int = 8000,
        num_samples_per_chunk = 7,
    ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    texts = text_splitter.split_text(original_text)
    print(f"Split the document into {len(texts)} parts")

    questions = []
    answers = []
    for text in texts:
        prompt = f"""Given  a document. Suggest {num_samples_per_chunk} questions could be asked related to the document. Generate a comprehensive and informative answer (but no more than 80 words) for each question.
            Document: {text}
            """

        messages=[
            {"role": "user", "content": prompt},
        ]

        outputs = get_completion_from_chatgpt(messages, openai_key)
        pairs = outputs.split("\n\n")
        for pair in pairs:
            question, answer = pair.split("\n")
            questions.append(question[3:])
            answers.append(answer)
    assert len(questions) == len(answers)
    return questions, answers

def prepare_seed_tasks(data_path, seed_path, api_key, chunk_size=8000, num_samples_per_chunk=7):
    instructions = []
    outputs = []
    for file in os.listdir(data_path):
        if file[-4:] != '.txt':
            continue
        with open(os.path.join(data_path, file)) as f:
            text = f.read()
        
        pairs = instruction_input_suggest(text, api_key, chunk_size, num_samples_per_chunk)
        instructions.extend(pairs[0])
        outputs.extend(pairs[1])
        #break 
    with open(seed_path, "a") as f:
        for i, instruction in enumerate(instructions):
            f.write(
                    json.dumps(
                        {
                            "id": f"seed_task_{i}",
                            "instruction": instruction,
                            "instances": [{"input":"", "output":outputs[i]}]
                        }
                    )
                    + "\n"
                )
