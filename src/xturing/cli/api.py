from pathlib import Path
from typing import List, Optional, Union

import click
import pydantic
import uvicorn
from fastapi import FastAPI

from xturing import BaseModel


class Params(pydantic.BaseModel):
    penalty_alpha: Optional[float] = 0.6
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    do_sample: Optional[bool] = False
    max_new_tokens: Optional[int] = 256


class UserInput(pydantic.BaseModel):
    prompt: Union[str, List[str]]
    params: Params


app = FastAPI()
model = None


@app.get("/health")
def health():
    return {"success": True, "message": "API server is running"}


@app.post("/api")
def xturing_api(user_input: UserInput):
    try:
        generation_config = model.generation_config()
        generation_config.penalty_alpha = user_input.params.penalty_alpha
        generation_config.top_k = user_input.params.top_k
        generation_config.top_p = user_input.params.top_p
        generation_config.do_sample = user_input.params.do_sample
        generation_config.max_new_tokens = user_input.params.max_new_tokens

        output = model.generate(texts=[user_input.prompt])

        return {"success": True, "response": output}

    except Exception as e:
        return {"success": False, "message": str(e)}


@click.command(name="api")
@click.option("-m", "--model_path")
def api_command(model_path: str):
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
