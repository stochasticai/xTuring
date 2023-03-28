from dataclasses import dataclass


class BaseOpenaiModel:
    @property
    def name(self):
        raise ValueError("You need to use specific model")


class Davinci(BaseOpenaiModel):
    name: str = "davinci"


class Ada(BaseOpenaiModel):
    name: str = "ada"
