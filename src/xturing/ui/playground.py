import time

import gradio as gr

from xturing.models.base import BaseModel


class Playground:
    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = model_path
        # load the model
        # self.model = BaseModel.create(weight_path=self.model_path)

    def launch(self) -> None:
        with gr.Blocks() as demo:
            with gr.Row():
                text1 = gr.Textbox(label="t1")

            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    text1 = gr.Textbox(label="prompt 1")
                with gr.Column(scale=2, min_width=600):
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox()
                    clear = gr.Button("Clear")

                    def user(user_message, history):
                        return "", history + [[user_message, None]]

                    def model(history):
                        # Pass user input to the model
                        # model_output = model.generate(texts=[history[-1][0]])

                        model_output = history[-1][0]
                        history[-1][1] = model_output
                        time.sleep(1)
                        return history

                    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                        model, chatbot, chatbot
                    )

                    clear.click(lambda: None, None, chatbot, queue=False)
        demo.launch()
