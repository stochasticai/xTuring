from pathlib import Path
from typing import Union

import ipywidgets as widgets
from ipywidgets import Layout

from xturing import BaseModel
from xturing.config import IS_INTERACTIVE


def model_chat(model_name_or_path: Union[str, Path]):
    assert IS_INTERACTIVE, "This function can only be run in a Jupyter Notebook"

    wrapped_model_path = Path(model_name_or_path)
    if wrapped_model_path.is_dir():
        model = BaseModel.load(str(wrapped_model_path))
    else:
        try:
            model = BaseModel.create(model_name_or_path)
        except:
            raise ValueError(
                "The model_name_or_path you have provided {model_name_or_path} is not valid"
            )

    def on_chat(b):
        with output:
            model_output = model.generate(texts=[ask.value])[0]
            print(model_output)

    def on_clear(b):
        with output:
            output.clear_output()

    ask = widgets.Text(
        value="",
        placeholder="Enter the TEXT",
        description="Prompt:",
        layout=Layout(width="100%", height="40px"),
        disabled=False,
    )
    chat_button = widgets.Button(description="Chat")
    clear_button = widgets.Button(description="Clear")
    output = widgets.Output()
    display(ask, output, chat_button, clear_button)

    chat_button.on_click(on_chat)
    clear_button.on_click(on_clear)
