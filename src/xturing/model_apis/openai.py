import time
from datetime import datetime

import openai

from xturing.model_apis.base import TextGenerationAPI


class OpenAITextGenerationAPI(TextGenerationAPI):
    config_name = "openai"

    def __init__(self, engine, api_key, organization, request_batch_size=10):
        super().__init__(
            engine=engine,
            api_key=api_key,
            organization=organization,
            request_batch_size=request_batch_size,
        )
        if api_key is not None:
            openai.api_key = api_key
        if organization is not None:
            openai.organization = organization
        self.request_batch_size = request_batch_size

    def get_completion(
        self,
        prompts,
        target_length,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop_sequences,
        logprobs,
        n,
        best_of,
    ):
        return openai.Completion.create(
            engine=self.engine,
            prompt=prompts,
            max_tokens=target_length,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of,
        )

    def generate_text(
        self,
        prompts,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop_sequences,
        logprobs,
        n,
        best_of,
        retries=3,
    ):
        response = None
        target_length = max_tokens
        if self.api_key is not None:
            openai.api_key = self.api_key
        if self.organization is not None:
            openai.organization = self.organization
        retry_cnt = 0
        backoff_time = 30
        while retry_cnt <= retries:
            try:
                response = self.get_completion(
                    prompts=prompts,
                    target_length=target_length,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop_sequences=stop_sequences,
                    logprobs=logprobs,
                    n=n,
                    best_of=best_of,
                )
                break
            except openai.error.OpenAIError as e:
                print(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    target_length = int(target_length * 0.8)
                    print(f"Reducing target length to {target_length}, Retrying...")
                else:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                retry_cnt += 1

        if isinstance(prompts, list):
            results = []
            for j, prompt in enumerate(prompts):
                data = {
                    "prompt": prompt,
                    "response": {"choices": response["choices"][j * n : (j + 1) * n]}
                    if response
                    else None,
                    "created_at": str(datetime.now()),
                }
                results.append(data)
            return results
        else:
            data = {
                "prompt": prompts,
                "response": response,
                "created_at": str(datetime.now()),
            }
            return [data]


class Davinci(OpenAITextGenerationAPI):
    config_name = "openai_davinci"

    def __init__(self, api_key, organization=None, request_batch_size=10):
        super().__init__(
            "davinci",
            api_key=api_key,
            organization=organization,
            request_batch_size=request_batch_size,
        )


class ChatGPT(OpenAITextGenerationAPI):
    config_name = "openai_chat"

    def __init__(self, api_key, organization=None, request_batch_size=10):
        super().__init__(
            "gpt-3.5-turbo",
            api_key=api_key,
            organization=organization,
            request_batch_size=request_batch_size,
        )

    def get_completion(self, prompts, **kwargs):
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        completion = openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            temperature=0,
        )
        return completion["choices"][0]["message"]["content"]
