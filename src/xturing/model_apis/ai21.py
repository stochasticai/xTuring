import time
from datetime import datetime

import ai21

from xturing.model_apis.base import TextGenerationAPI


class AI21TextGenerationAPI(TextGenerationAPI):
    config_name = "ai21"

    def __init__(self, engine, api_key):
        super().__init__(engine, api_key=api_key, request_batch_size=1)
        ai21.api_key = api_key

    def generate_text(
        self,
        prompts,
        max_tokens,
        temperature,
        top_p,
        stop_sequences,
        retries=3,
        **kwargs,
    ):
        response = None
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= retries:
            try:
                response = ai21.Completion.execute(
                    model=self.engine,
                    prompt=prompts[0],
                    numResults=1,
                    maxTokens=max_tokens,
                    temperature=temperature,
                    topKReturn=0,
                    topP=top_p,
                    stopSequences=stop_sequences,
                )
                break
            except Exception as e:
                print(f"AI21Error: {e}.")
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1

        predicts = {
            "choices": [
                {
                    "text": response["prompt"]["text"],
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


class J2Grande(AI21TextGenerationAPI):
    config_name = "ai21_j2_grande"

    def __init__(self, api_key):
        super().__init__(engine="j2-grande", api_key=api_key)
