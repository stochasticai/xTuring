import time
from datetime import datetime

import cohere

from xturing.model_apis.base import TextGenerationAPI


class CohereTextGenerationAPI(TextGenerationAPI):
    config_name = "cohere"

    def __init__(self, engine, api_key):
        super().__init__(engine, api_key=api_key, request_batch_size=1)

    def generate_text(
        self,
        prompts,
        max_tokens,
        temperature,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop_sequences=None,
        logprobs=None,
        n=None,
        retries=3,
        **kwargs,
    ):
        response = None
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= retries:
            try:
                co = cohere.Client(self.api_key)
                response = co.generate(
                    model=self.engine,
                    prompt=prompts[0],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
                break
            except cohere.CohereError as e:
                print(f"CohereError: {e}.")
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1

        predicts = {
            "choices": [
                {
                    "text": response.generations[0].text,
                    "finish_reason": "eos",
                }
            ]
        }

        data = {
            "prompt": prompts,
            "response": predicts,
            "created_at": str(datetime.now()),
        }
        return [data]


class Medium(CohereTextGenerationAPI):
    config_name = "cohere_medium"

    def __init__(self, api_key):
        super().__init__("medium", api_key)
