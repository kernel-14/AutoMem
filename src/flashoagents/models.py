#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


from openai import (
    BadRequestError,
    APIStatusError,
    APIConnectionError,
    OpenAIError,
)
import time

from .tools import Tool
from .utils import encode_image_base64, make_image_url


logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
}


class EmptyContentError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"Empty content in response: {response}")



def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls]
        return cls(role=message.role, content=message.content, tool_calls=tool_calls, reasoning_content=message.reasoning_content, raw=raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            # ** ** ** ** ** ** ** *0307hhy ** ** ** ** *
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(
                        **{k: v for k, v in tc["function"].items() if k != "parameters"}
                    ),
                    id=tc["id"],
                    type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


class Model:
    def __init__(self, **kwargs):
        # Last call statistics (single API call)
        self.last_input_token_count = None
        self.last_output_token_count = None
        
        # Task-level cumulative statistics (multiple API calls in one task)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        
        self.kwargs = kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        # Opt-in reasoning-effort control (e.g. taiji hy3 defaults to slow
        # thinking, which dominates wall-clock on agentic loops). Env-gated so
        # behavior is unchanged unless explicitly requested. NOTE: this file is
        # covered by the engine's package_source_sha256 protocol digest -- any
        # change here invalidates resume for in-flight runs (deliberately
        # accepted for the no_think rerun on the webwalkerqa branch).
        _reasoning = os.environ.get("LLM_REASONING_EFFORT", "").strip()
        if _reasoning:
            extra_body = dict(completion_kwargs.get("extra_body") or {})
            extra_body.setdefault("reasoning_effort", _reasoning)
            completion_kwargs["extra_body"] = extra_body

        return completion_kwargs

    def reset_total_counts(self):
        """Reset cumulative token counts for a new task"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
    
    def get_token_counts(self) -> Dict[str, int]:
        """Get token counts from the last API call"""
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }
    
    def get_total_counts(self) -> Dict[str, int]:
        """Get cumulative token counts for the current task"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_api_calls": self.total_api_calls,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!

    def to_dict(self) -> Dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: Dict[str, Any]) -> "Model":
        model_instance = cls(
            **{
                k: v
                for k, v in model_dictionary.items()
                if k not in ["last_input_token_count", "last_output_token_count"]
            }
        )
        model_instance.last_input_token_count = model_dictionary.pop("last_input_token_count", None)
        model_instance.last_output_token_count = model_dictionary.pop("last_output_token_count", None)
        return model_instance


class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        import httpx
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=600.0),
        )
        self.custom_role_conversions = custom_role_conversions

    @staticmethod
    def truncate_content_based_on_stop_sequences(content: str, stop_sequences: List[str]) -> str:
        if not stop_sequences:
            return content
        # 在stop_seq之后截断content
        for stop_seq in stop_sequences:
            index = content.find(stop_seq)
            if index != -1:
                content = content[:index + len(stop_seq)]
                break  # Only keep the first match
        return content
    
    @staticmethod
    def remove_think_tags(content: str) -> str:
        """移除content中的think相关标签（<think>、<think>等）"""
        if not content:
            return content
        import re
        # 移除<think>...</think>标签及其内容（支持多行）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 移除<think>...</think>标签及其内容（支持多行）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 移除自闭合标签
        content = re.sub(r'<think\s*/>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<redacted_reasoning\s*/>', '', content, flags=re.IGNORECASE)
        return content.strip()

    def _stream_and_collect(self, completion_kwargs: dict):
        """Send a streaming request and reassemble into a non-streaming response object."""
        import openai
        completion_kwargs["stream"] = True
        completion_kwargs["stream_options"] = {"include_usage": True}
        stream = self.client.chat.completions.create(**completion_kwargs)

        content_parts = []
        reasoning_parts = []
        tool_calls_map: dict = {}  # index -> {id, type, function:{name, arguments}}
        role = "assistant"
        model = ""
        resp_id = ""
        created = 0
        finish_reason = None
        usage = None

        for chunk in stream:
            if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                continue
            if not chunk.choices:
                continue
            resp_id = chunk.id or resp_id
            model = chunk.model or model
            created = chunk.created or created
            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if delta is None:
                continue
            if delta.role:
                role = delta.role
            if delta.content:
                content_parts.append(delta.content)
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": tc.function.name or "", "arguments": ""},
                        }
                    else:
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function.name:
                            tool_calls_map[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_map[idx]["function"]["arguments"] += tc.function.arguments
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

        # Build a fake non-streaming response
        full_content = "".join(content_parts) or None
        full_reasoning = "".join(reasoning_parts) or None
        tool_calls_list = None
        if tool_calls_map:
            tool_calls_list = []
            for idx in sorted(tool_calls_map.keys()):
                tc = tool_calls_map[idx]
                tool_calls_list.append(
                    openai.types.chat.chat_completion_message.ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=openai.types.chat.chat_completion_message.Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

        message = openai.types.chat.ChatCompletionMessage(
            role=role,
            content=full_content,
            tool_calls=tool_calls_list,
            refusal=None,
        )
        if full_reasoning is not None:
            message.reasoning_content = full_reasoning

        if usage is None:
            # Fallback usage estimate
            usage = openai.types.CompletionUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )

        choice = openai.types.chat.chat_completion.Choice(
            index=0,
            message=message,
            finish_reason=finish_reason or "stop",
        )

        return openai.types.chat.ChatCompletion(
            id=resp_id or "stream-collected",
            choices=[choice],
            created=created or 0,
            model=model or completion_kwargs.get("model", ""),
            object="chat.completion",
            usage=usage,
        )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        # Task-level token cap (2026-07-07, opt-in). Enforced ONLY on the task
        # agent's model (process_item sets `_enforce_token_cap=True` after
        # reset_total_counts) — NOT on long-lived meta models (proposer / judge /
        # extraction) which accumulate across the whole run and would otherwise
        # be wrongly aborted. Checked BEFORE the next call so a runaway task stops
        # growing instead of blowing the model context. The raise is caught by
        # process_item's per-task except (records error/status="error") and
        # attribution routes it to the out-of-scope INFRA_ERROR bucket.
        if getattr(self, "_enforce_token_cap", False):
            _cap = os.environ.get("TASK_TOKEN_CAP")
            try:
                _cap_n = int(_cap) if _cap else 0
            except (TypeError, ValueError):
                _cap_n = 0
            if _cap_n > 0 and (self.total_input_tokens + self.total_output_tokens) >= _cap_n:
                raise RuntimeError(
                    f"[task_token_cap] cumulative task tokens "
                    f"{self.total_input_tokens + self.total_output_tokens} >= cap {_cap_n}; "
                    "aborting task (out-of-scope infra limit)."
                )
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Check if model_id contains 'o3' or 'o4'
        if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
            # Remove stop_sequences from completion_kwargs
            completion_kwargs.pop('stop', None)

        # response = self.client.chat.completions.create(**completion_kwargs)

        # Use streaming if FORCE_STREAM env var is set (required by some API proxies)
        use_stream = os.environ.get("FORCE_STREAM", "").strip().lower() in ("1", "true", "yes")

        max_retries = 5
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                if use_stream:
                    response = self._stream_and_collect(completion_kwargs)
                else:
                    response = self.client.chat.completions.create(**completion_kwargs)

                # Update last call counts
                usage = response.usage
                if usage is not None:
                    self.last_input_token_count = usage.prompt_tokens or 0
                    self.last_output_token_count = usage.completion_tokens or 0
                    self.total_input_tokens += self.last_input_token_count
                    self.total_output_tokens += self.last_output_token_count
                else:
                    self.last_input_token_count = 0
                    self.last_output_token_count = 0
                self.total_api_calls += 1

                if not getattr(response, "choices", None):
                    raise EmptyContentError(response)

                first_choice = response.choices[0]
                message_obj = getattr(first_choice, "message", None)
                if message_obj is None:
                    raise EmptyContentError(response)

                if not message_obj.content and not getattr(message_obj, 'tool_calls', None):  # o1 o3-mini
                    raise EmptyContentError(response)

                if hasattr(message_obj, "model_dump"):
                    message_payload = message_obj.model_dump(
                        include={"role", "content", "tool_calls", "reasoning_content"}
                    )
                else:
                    message_payload = {
                        "role": getattr(message_obj, "role", "assistant"),
                        "content": getattr(message_obj, "content", None),
                        "tool_calls": getattr(message_obj, "tool_calls", None),
                        "reasoning_content": getattr(message_obj, "reasoning_content", None),
                    }

                message = ChatMessage.from_dict(message_payload)
                message.raw = response

                # If model_id contains 'o3' or 'o4', manually truncate content based on stop_sequences
                if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
                    message.content = self.truncate_content_based_on_stop_sequences(message.content, stop_sequences)

                # 移除content中的think标签（适用于所有带think标签的模型）
                if message.content:
                    message.content = self.remove_think_tags(message.content)

                if tools_to_call_from is not None:
                    return parse_tool_args_if_needed(message)
                return message
            except BadRequestError as e:
                logger.error(f"Bad Request Error: {e}")
                raise
            except APIConnectionError as e:
                if attempt < max_retries:
                    logging.warning(f"Network error occurred: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except (APIStatusError, EmptyContentError) as e:
                if attempt < max_retries:
                    logging.warning(f"API status error occurred: {e}. Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except OpenAIError as e:
                logging.error(f"API error occurred: {e}.")
                raise
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}.")
                raise

__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "OpenAIServerModel",
    "ChatMessage",
]
