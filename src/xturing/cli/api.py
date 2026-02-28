import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

import click
import pydantic
import uvicorn
from fastapi import FastAPI, HTTPException


class Params(pydantic.BaseModel):
    penalty_alpha: Optional[float] = 0.6
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    do_sample: Optional[bool] = False
    max_new_tokens: Optional[int] = 256


class UserInput(pydantic.BaseModel):
    prompt: Union[str, List[str]]
    params: Optional[Params] = None


class OpenAIChatMessage(pydantic.BaseModel):
    role: str
    content: str


class OpenAIChatCompletionRequest(pydantic.BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1


app = FastAPI()
model = None


def _loaded_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def _as_prompt_list(prompt: Union[str, List[str]]) -> List[str]:
    return prompt if isinstance(prompt, list) else [prompt]


def _apply_generation_params(target_model, params: Params):
    generation_config = target_model.generation_config()
    generation_config.penalty_alpha = params.penalty_alpha
    generation_config.top_k = params.top_k
    generation_config.top_p = params.top_p
    generation_config.do_sample = params.do_sample
    generation_config.max_new_tokens = params.max_new_tokens


def _conversation_to_prompt(messages: List[OpenAIChatMessage]) -> str:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    return "\n".join([f"{message.role}: {message.content}" for message in messages])


@app.get("/health")
def health():
    return {"success": True, "message": "API server is running"}


@app.post("/api")
def xturing_api(user_input: UserInput):
    try:
        active_model = _loaded_model()
        params = user_input.params or Params()
        _apply_generation_params(active_model, params)
        output = active_model.generate(texts=_as_prompt_list(user_input.prompt))

        return {"success": True, "response": output}

    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/v1/models")
def list_models():
    active_model = _loaded_model()
    model_id = getattr(active_model, "model_name", "xturing-model")
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}


@app.post("/v1/chat/completions")
def openai_chat_completions(user_input: OpenAIChatCompletionRequest):
    active_model = _loaded_model()
    params = Params()
    if user_input.temperature is not None:
        params.do_sample = user_input.temperature > 0
        if params.do_sample:
            params.penalty_alpha = None
    if user_input.top_p is not None:
        params.top_p = user_input.top_p
        params.do_sample = True
        params.penalty_alpha = None
    if user_input.max_tokens is not None:
        params.max_new_tokens = user_input.max_tokens

    _apply_generation_params(active_model, params)

    prompt = _conversation_to_prompt(user_input.messages)
    output = active_model.generate(texts=[prompt])[0]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    model_id = user_input.model or getattr(active_model, "model_name", "xturing-model")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
    }


@click.command(name="api")
@click.option(
    "-m",
    "--model_path",
    required=True,
    help="Path to a model directory containing xturing.json",
)
def api_command(model_path: str):
    from xturing import BaseModel

    # Resolve the path
    wrapped_model_path = Path(model_path)

    # Check if the user provide model path is a directory
    if wrapped_model_path.is_dir():
        click.secho("[*] Loading your model...", fg="blue", bold=True)
        global model
        model = BaseModel.load(str(wrapped_model_path))

    else:
        click.secho(
            f"[-] The model_path you have provided {model_path} is not valid",
            fg="red",
            bold=True,
        )
        return

    click.secho("[+] Model loaded successfully.", fg="green", bold=True)

    uvicorn.run(app, port=5000, workers=1)
