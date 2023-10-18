from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional


@dataclass
class ChatMessage:
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"

    content: str
    role: str
    name: Optional[str]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict, hash=False)

    def is_from(self, role: Literal["assistant", "user", "system", "function"]) -> bool:
        return self.role == role

    @classmethod
    def from_assistant(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.ASSISTANT, None)

    @classmethod
    def from_user(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.USER, None)

    @classmethod
    def from_system(cls, content: str) -> "ChatMessage":
        return cls(content, ChatMessage.SYSTEM, None)

    @classmethod
    def from_function(cls, content: str, name: str) -> "ChatMessage":
        return cls(content, ChatMessage.FUNCTION, name)
