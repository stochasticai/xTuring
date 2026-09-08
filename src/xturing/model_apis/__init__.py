from xturing.model_apis.base import BaseApi, TextGenerationAPI
from xturing.model_apis.claude import ClaudeSonnet, ClaudeTextGenerationAPI
from xturing.model_apis.cohere import CohereTextGenerationAPI
from xturing.model_apis.cohere import Medium as CohereMedium
from xturing.model_apis.openai import ChatGPT as OpenAIChatGPT
from xturing.model_apis.openai import Davinci as OpenAIDavinci
from xturing.model_apis.openai import OpenAITextGenerationAPI
from xturing.model_apis.qwen import Qwen3OmniTextGenerationAPI

BaseApi.add_to_registry(OpenAITextGenerationAPI.config_name, OpenAITextGenerationAPI)
BaseApi.add_to_registry(CohereTextGenerationAPI.config_name, CohereTextGenerationAPI)
BaseApi.add_to_registry(ClaudeTextGenerationAPI.config_name, ClaudeTextGenerationAPI)
BaseApi.add_to_registry(OpenAIDavinci.config_name, OpenAIDavinci)
BaseApi.add_to_registry(OpenAIChatGPT.config_name, OpenAIChatGPT)
BaseApi.add_to_registry(CohereMedium.config_name, CohereMedium)
BaseApi.add_to_registry(ClaudeSonnet.config_name, ClaudeSonnet)
BaseApi.add_to_registry(
    Qwen3OmniTextGenerationAPI.config_name, Qwen3OmniTextGenerationAPI
)
