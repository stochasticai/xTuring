from typing import Dict, List


class ListPromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: List[str],
        list_templates: Dict[str, str] = None,
    ):
        self.template = template
        self.input_variables = input_variables

        self.list_templates = (
            list_templates  # key words in the list template are number and text
        )
        if self.list_templates is None:
            self.list_templates = {}

    def check_list_template(self, list_template: str):
        return list_template in self.list_templates

    @classmethod
    def process_list_template(cls, inputs: List[str], list_template: str):
        return "\n".join(
            list_template.format(number=i, text=text) for i, text in enumerate(inputs)
        )

    def build(self, **kwargs) -> str:
        for i in self.input_variables:
            if i not in kwargs:
                raise ValueError(f"Missing input variable {i}")

        for k, v in kwargs.items():
            if isinstance(v, list):
                if k not in self.list_templates:
                    raise ValueError(f"Missing list template for variable {k}")
                kwargs[k] = self.process_list_template(v, self.list_templates[k])

        return self.template.format(**kwargs)
