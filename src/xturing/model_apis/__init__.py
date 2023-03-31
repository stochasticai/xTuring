from .ai21 import AI21TextGenerationAPI
from .base import BaseApi, TextGenerationAPI
from .cohere import CohereTextGenerationAPI
from .openai import Davinci, OpenAITextGenerationAPI

BaseApi.add_to_registry(OpenAITextGenerationAPI.config_name, OpenAITextGenerationAPI)
BaseApi.add_to_registry(Davinci.config_name, Davinci)
BaseApi.add_to_registry(CohereTextGenerationAPI.config_name, CohereTextGenerationAPI)
BaseApi.add_to_registry(AI21TextGenerationAPI.config_name, AI21TextGenerationAPI)
