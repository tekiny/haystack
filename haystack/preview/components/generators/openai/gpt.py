import json
from typing import Optional, List, Callable, Dict, Any, Union

import sys
import logging
from collections import defaultdict
import dataclasses
import openai
from openai.openai_object import OpenAIObject

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError
from haystack.preview.dataclasses.chat_message import ChatMessage

logger = logging.getLogger(__name__)


API_BASE_URL = "https://api.openai.com/v1"


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from OpenAI API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, flush=True, end="")
    return chunk


@component
class GPTGenerator:
    """
    LLM Generator compatible with GPT (ChatGPT) large language models.

    Queries the LLM using OpenAI's API. Invocations are made using OpenAI SDK ('openai' package)
    See [OpenAI GPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Creates an instance of GPT35Generator for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name of the model to use.
        :param system_prompt: An additional message to be sent to the LLM at the beginning of each conversation.
            Typically, a conversation is formatted with a system message first, followed by alternating messages from
            the 'user' (the "queries") and the 'assistant' (the "responses"). The system message helps set the behavior
            of the assistant. For example, you can modify the personality of the assistant or provide specific
            instructions about how it should behave throughout the conversation.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the OpenAI
            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model_parameters = kwargs
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        if self.streaming_callback:
            module = self.streaming_callback.__module__
            if module == "builtins":
                callback_name = self.streaming_callback.__name__
            else:
                callback_name = f"{module}.{self.streaming_callback.__name__}"
        else:
            callback_name = None

        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            **self.model_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        streaming_callback = None
        if "streaming_callback" in init_params and init_params["streaming_callback"]:
            parts = init_params["streaming_callback"].split(".")
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
            streaming_callback = getattr(module, function_name, None)
            if not streaming_callback:
                raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
            data["init_parameters"]["streaming_callback"] = streaming_callback
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: Union[str, List[ChatMessage]]):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompt: The prompts to be sent to the generative model.
        """
        messages: List[ChatMessage] = []
        if isinstance(prompt, str):
            message = ChatMessage.from_user(prompt)
            if self.system_prompt:
                messages = [ChatMessage.from_system(self.system_prompt), message]
            else:
                messages = [message]
        elif isinstance(prompt, list) and all(isinstance(message, ChatMessage) for message in prompt):
            messages = prompt
        else:
            raise ValueError(
                f"Invalid prompt. Expected either a string or a list of ChatMessage(s), but got {type(prompt)}"
            )
        openai_chat_message_format = ["role", "content", "name"]
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            api_key=self.api_key,
            messages=[
                dataclasses.asdict(
                    m, dict_factory=lambda obj: {k: v for k, v in obj if k in openai_chat_message_format and v}
                )
                for m in messages
            ],
            stream=self.streaming_callback is not None,
            **self.model_parameters,
        )

        replies: List[ChatMessage] = [self._build_response(completion, choice) for choice in completion.choices]
        return {"replies": replies}

    def _build_response(self, completion: OpenAIObject, choice: OpenAIObject) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.
        """
        message: ChatMessage
        if choice.finish_reason == "function_call":
            message = ChatMessage.from_assistant(content=json.loads(str(choice.message.function_call)))
        else:
            message = ChatMessage.from_assistant(content=choice.message.content)
        message.metadata.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage.items()),
            }
        )
        return message
