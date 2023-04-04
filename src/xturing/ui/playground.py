import gradio as gr

from xturing.models.base import BaseModel

model_to_class_map = {"GPT-2": "gpt2", "GPT-J": "gptj", "Llama": "llama"}


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

    def set_model(self, model_path=None, model_name=None):
        print(f"model_path:{model_path}, model_name: {model_name}")

        try:
            if model_path:
                self.model = BaseModel.load(model_path)
                return True, ""
            elif model_name:
                self.model = BaseModel.create(model_to_class_map[model_name])
                return True, ""
            else:
                return False, "Model path is required."
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return False, str(e)

    def generate_output(self, user_input):
        if user_input == "":
            return "Enter a valid prompt"

        generation_config = self.model.generation_config()
        generation_config.penalty_alpha = self.penalty_alpha
        generation_config.top_k = self.top_k
        generation_config.top_p = self.top_p
        generation_config.do_sample = self.do_sample
        generation_config.max_new_tokens = self.max_new_tokens

        print(f"Prompt:{user_input}, Params:{str(generation_config)}")

        try:
            return self.model.generate(texts=[user_input])
        except Exception as e:
            print(str(e))
            return "Error generating output. Please try again."

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

            with gr.Row():
                # LEFT COLUMN
                with gr.Column(scale=3, min_width=600):
                    # model_selection_type = gr.Radio(
                    #     label="How do you want to load the model?",
                    #     choices=["Finetuned model path", "Baseline model"],
                    #     value="Finetuned model path",
                    # )

                    model_path_input = gr.Textbox(
                        value=self.model_path if self.model_path else "",
                        interactive=True,
                        placeholder="Enter the model path",
                        label="Model path",
                        visible=True,
                    )

                    # baseline_models_dropdown = gr.Dropdown(
                    #     interactive=True,
                    #     visible=False,
                    #     choices=["GPT-2", "GPT-J", "Llama"],
                    #     label="Select model",
                    # )

                    def load_func(model_path, model_name):
                        success, message = self.set_model(model_path, model_name)

                        if success:
                            print("Model loaded successfully")
                            return (
                                gr.update(
                                    visible=True,
                                    value="""
                            <h3 style="color:green;text-align:center">Model loaded successfully</h3>
                            """,
                                ),
                                gr.update(
                                    interactive=True,
                                    placeholder="Enter your prompt and press enter",
                                ),
                                gr.update(),
                            )
                        else:
                            return (
                                gr.update(
                                    visible=True,
                                    value="""<h3 style="color:red;text-align:center;word-break: break-all;">Model load failed:{}{}</h3>""".format(
                                        " ", message
                                    ),
                                ),
                                gr.update(),
                                gr.update(),
                            )

                    load_model_btn = gr.Button("Load")
                    load_model_error = gr.Markdown(visible=False, value="")

                    # def update_model_loading_input(model_selection_type):
                    #     if model_selection_type == "Finetuned model path":
                    #         return (
                    #             gr.update(visible=True, value=""),
                    #             gr.update(visible=False, value=""),
                    #             gr.update(visible=False),
                    #         )
                    #     else:
                    #         return (
                    #             gr.update(visible=False, value=""),
                    #             gr.update(visible=True, value="GPT-2"),
                    #             gr.update(visible=False),
                    #         )

                    # model_selection_type.change(
                    #     update_model_loading_input,
                    #     model_selection_type,
                    #     [model_path_input, baseline_models_dropdown, load_model_error],
                    # )

                    gr.Markdown(
                        """
                        # Chat with your model here
                        """
                    )

                    clear = gr.Button("Clear chat", show_label=False)
                    chatbot = gr.Chatbot(label="Chat with your model")
                    msg = gr.Textbox(
                        label="Prompt",
                        interactive=False,
                        placeholder="Load a model to enable the chat",
                    )

                    load_model_btn.click(
                        load_func,
                        inputs=[
                            model_path_input,
                            # baseline_models_dropdown,
                        ],
                        outputs=[load_model_error, msg, load_model_btn],
                        show_progress=True,
                    )

                    def user(user_message, history):
                        return "", history + [[user_message, None]]

                    def model(history):
                        # Pass user input to the model
                        model_output = "🤖 : " + self.generate_output(history[-1][0])[0]
                        history[-1][1] = model_output
                        return history

                    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                        model, chatbot, chatbot
                    )

                    clear.click(lambda: None, None, chatbot, queue=False)

                # RIGHT COLUMN
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
                            info="Choose between 0 and 1",
                        )
                        penalty_alpha.release(
                            lambda penalty_alpha: self.set_penalty_alpha(penalty_alpha),
                            penalty_alpha,
                        )
                        top_k = gr.Slider(
                            1,
                            40,
                            value=4,
                            label="Top-k",
                            interactive=True,
                            info="Choose between 1 and 40",
                        )
                        top_k.release(lambda top_k: self.set_top_k(top_k), top_k)
                        max_new_tokens = gr.Slider(
                            1,
                            512,
                            value=self.max_new_tokens,
                            label="Max new tokens",
                            interactive=True,
                            info="Choose between 1 and 512",
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
                            label="Top-k",
                            interactive=False,
                            info="Always zero",
                            visible=False,
                        )
                        top_k.release(lambda top_k: self.set_top_k(top_k), top_k)

                        top_p = gr.Slider(
                            0,
                            1,
                            value=0.92,
                            label="Top-p",
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
                            info="Choose between 1 and 512",
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
                            self.set_top_k(4)
                            self.set_do_sample(False)
                            self.set_top_p(1)
                            self.set_penalty_alpha(0.6)
                            return (
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(value=4),
                                gr.update(value=1),
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
                        show_progress=True,
                    )

        demo.launch()
