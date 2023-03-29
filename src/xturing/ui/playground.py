import gradio as gr

from xturing.models.base import BaseModel


class Playground:
    def __init__(
        self,
        model_path: str = None,
    ):
        self.penalty_alpha = None
        self.top_k = None
        self.top_p = 0.92
        self.do_sample = True
        self.max_new_tokens = 256

        self.model_path = model_path
        self.model = None

        # load the model
        # self.model = BaseModel.load(self.model_path) if model_path else None
        # self.generation_config = self.model.generation_config()

    def set_model(self, model_path=None, model_name=None):
        print(f"model_path:{model_path}, model_name: {model_name}")
        # if model_path and model_path != "":
        #     self.model = BaseModel.load(model_path)
        # elif model_name and model_name != "":
        #     self.model = BaseModel.create(model_name)

    def generate_output(self, user_input):
        generation_config = self.model.generation_config()
        generation_config.penalty_alpha = self.penalty_alpha
        generation_config.top_k = self.top_k
        generation_config.top_p = self.top_p
        generation_config.do_sample = self.do_sample
        generation_config.max_new_tokens = self.max_new_tokens

        return self.model.generate(texts=[user_input])

    def set_penalty_alpha(self, penalty_alpha):
        self.penalty_alpha = penalty_alpha

    def set_top_k(self, top_k):
        self.top_k = int(top_k) if top_k else None

    def set_top_p(self, top_p):
        self.top_p = top_p

    def set_do_sample(self, do_sample):
        self.do_sample = do_sample

    def set_max_new_tokens(self, max_new_tokens):
        self.max_new_tokens = max_new_tokens

    # utility function
    # def update_model_path(self, path):

    def launch(self) -> None:
        with gr.Blocks() as demo:
            gr.Markdown(
                """
            # xTuring playground
            """
            )

            # update_model_path.click()
            with gr.Row():
                with gr.Column(scale=3, min_width=600):
                    model_selection_type = gr.Radio(
                        label="How do you want to load the model?",
                        choices=["Finetuned model path", "Baseline model"],
                        value="Finetuned model path",
                    )

                    model_path_input = gr.Textbox(
                        value="",
                        interactive=True,
                        placeholder="Enter the model path",
                        label="Enter the model path",
                        visible=True,
                    )

                    baseline_models_dropdown = gr.Dropdown(
                        interactive=True,
                        visible=False,
                        choices=["GPT-2", "GPT-J", "Llama"],
                        label="Select model",
                    )

                    load_model_btn = gr.Button("Load")
                    load_model_btn.click(
                        lambda model_path, model_name: self.set_model(
                            model_path, model_name
                        ),
                        inputs=[model_path_input, baseline_models_dropdown],
                    )

                    def update_model_loading_input(model_selection_type):
                        if model_selection_type == "Finetuned model path":
                            return gr.update(visible=True, value=""), gr.update(
                                visible=False, value=""
                            )
                        else:
                            return gr.update(visible=False, value=""), gr.update(
                                visible=True, value="GPT-2"
                            )

                    model_selection_type.change(
                        update_model_loading_input,
                        model_selection_type,
                        [model_path_input, baseline_models_dropdown],
                    )

                    gr.Markdown(
                        """
                        # Chat with your model here
                        """
                    )

                    chatbot = gr.Chatbot(label="Chat with your model")
                    msg = gr.Textbox(
                        label="Prompt",
                        interactive=self.model != None,
                        placeholder="Load a model to enable the chat"
                        if self.model == None
                        else "Enter your prompt here",
                    )
                    clear = gr.Button("Clear", show_label=False)

                    def user(user_message, history):
                        return "", history + [[user_message, None]]

                    def model(history):
                        # Pass user input to the model
                        # model_output = self.generate_output(history[-1][0])
                        model_output = "🤖" + history[-1][0]
                        history[-1][1] = model_output
                        return history

                    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                        model, chatbot, chatbot
                    )

                    clear.click(lambda: None, None, chatbot, queue=False)

                with gr.Column(scale=1, min_width=600):
                    # All the parameters
                    decoding_method_radio = gr.Radio(
                        label="Decoding method",
                        choices=["Top-p sampling", "Contrastive search"],
                        value="Top-p sampling",
                    )

                    gr.Markdown(
                        """
                        # Parameters
                        """
                    )

                    with gr.Column(visible=False) as contrstive_search_column:
                        penalty_alpha = gr.Slider(
                            0,
                            1,
                            value=0.6,
                            label="Penalty alpha",
                            interactive=True,
                            info="Choose betwen 0 and 1",
                        )
                        penalty_alpha.release(
                            lambda penalty_alpha: self.set_penalty_alpha(penalty_alpha),
                            penalty_alpha,
                        )
                        top_k = gr.Slider(
                            1,
                            40,
                            value=4,
                            label="TopK",
                            interactive=True,
                            info="Choose betwen 1 and 40",
                        )
                        top_k.release(lambda top_k: self.set_top_k(top_k), top_k)
                        max_new_tokens = gr.Slider(
                            1,
                            512,
                            value=self.max_new_tokens,
                            label="Max new tokens",
                            interactive=True,
                            info="Choose betwen 1 and 512",
                        )
                        max_new_tokens.release(
                            lambda max_new_tokens: self.set_max_new_tokens(
                                max_new_tokens
                            ),
                            max_new_tokens,
                        )

                    with gr.Column(visible=True) as top_p_sampling_column:
                        top_k = gr.Slider(
                            0,
                            40,
                            value=0,
                            label="TopK",
                            interactive=False,
                            info="Always zero",
                            visible=False,
                        )
                        top_k.release(lambda top_k: self.set_top_k(top_k), top_k)

                        top_p = gr.Slider(
                            0,
                            1,
                            value=0.92,
                            label="TopP",
                            interactive=True,
                            info="Choose between 0 and 1",
                        )
                        top_p.release(lambda top_p: self.set_top_p(top_p), top_p)

                        gr.Radio(
                            label="Do sample",
                            interactive=False,
                            choices=[True, False],
                            value=True,
                            visible=False,
                        )
                        max_new_tokens2 = gr.Slider(
                            1,
                            512,
                            value=self.max_new_tokens,
                            label="Max new tokens",
                            interactive=True,
                            info="Choose betwen 1 and 512",
                        )
                        max_new_tokens2.release(
                            lambda max_new_tokens: self.set_max_new_tokens(
                                max_new_tokens
                            ),
                            max_new_tokens2,
                        )

                    def handle_decoding_method_change(decoding_method_radio):
                        if decoding_method_radio == "Top-p sampling":
                            self.set_top_k(0)
                            self.set_do_sample(True)
                            self.set_top_p(0.92)
                            self.set_penalty_alpha(None)
                            return (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(value=0),
                                gr.update(value=0.92),
                                gr.update(value=0),
                                gr.update(value=self.max_new_tokens),
                                gr.update(value=self.max_new_tokens),
                            )
                        else:
                            self.set_top_k(None)
                            self.set_do_sample(None)
                            self.set_top_p(None)
                            self.set_penalty_alpha(0.6)
                            return (
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(value=0),
                                gr.update(value=0),
                                gr.update(value=0.6),
                                gr.update(value=self.max_new_tokens),
                                gr.update(value=self.max_new_tokens),
                            )

                    decoding_method_radio.change(
                        handle_decoding_method_change,
                        decoding_method_radio,
                        [
                            top_p_sampling_column,
                            contrstive_search_column,
                            top_k,
                            top_p,
                            penalty_alpha,
                            max_new_tokens,
                            max_new_tokens2,
                        ],
                    )

        demo.launch()
