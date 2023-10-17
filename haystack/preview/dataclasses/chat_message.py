from dataclasses import dataclass
from typing import Literal


@dataclass
class ChatMessage:
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"

    content: str
    role: str

    def is_from(self, role: Literal["assistant", "user", "system", "function"]) -> bool:
        return self.role == role

    @classmethod
    def from_assistant(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.ASSISTANT)

    @classmethod
    def from_user(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.USER)

    @classmethod
    def from_system(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.SYSTEM)

    @classmethod
    def from_function(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.FUNCTION)
