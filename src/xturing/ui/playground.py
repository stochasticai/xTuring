import random
import time

import gradio as gr

from xturing.models.base import BaseModel


class Playground:
    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = model_path

    def launch(self) -> None:
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def model(history):
                # load the model
                # model = BaseModel.create(weight_path=self.model_path)
                # model_output = model.generate(texts=["Why LLM models are becoming so important?"])
                # model_output = random.choice(["Yes", "No"])
                model_output = history[-1][0]
                history[-1][1] = model_output
                time.sleep(1)
                return history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                model, chatbot, chatbot
            )

            clear.click(lambda: None, None, chatbot, queue=False)

        demo.launch()
