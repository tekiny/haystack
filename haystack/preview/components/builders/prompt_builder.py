from typing import Dict, Any, Optional, List

from jinja2 import Template, meta

from haystack.preview import component
from haystack.preview import default_to_dict, default_from_dict
from haystack.preview.dataclasses.chat_message import ChatMessage


@component
class PromptBuilder:
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 engine.
    The template variables found in the template string are used as input types for the component and are all required.

    Usage:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```
    """

    def __init__(self, template: Optional[str] = None, template_variables: Optional[List[str]] = None):
        """
        Initialize the component with a template string. If template_variables are not provided, the component will
        parse the template string and use the template variables as input types.

        :param template: Template string to be rendered.
        :param template_variables: List of template variables to be used as input types.
        """
        if template_variables:
            dynamic_input_types = {var: Any for var in template_variables}
        else:
            if not template:
                raise ValueError("Either template or template_variables must be provided.")
            self._template_string = template
            self.template = Template(template)
            ast = self.template.environment.parse(template)
            template_variables = meta.find_undeclared_variables(ast)
            dynamic_input_types = {var: Any for var in template_variables}

        static_input_type = {"messages": Optional[List[ChatMessage]]}
        component.set_input_types(self, **static_input_type, **dynamic_input_types)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, template=self._template_string)

    @classmethod
    def from_dict(cls, data) -> "PromptBuilder":
        return default_from_dict(cls, data)

    @component.output_types(prompt=str)
    def run(self, messages: Optional[List[ChatMessage]] = None, **kwargs):
        if messages:
            last_message: ChatMessage = messages[-1]
            if last_message.is_from(ChatMessage.USER):
                template = Template(last_message.content)
                messages[-1] = ChatMessage.from_user(template.render(kwargs))
            return {"prompt": messages}
        else:
            return {"prompt": self.template.render(kwargs)}
