"""AppWorld agent adapter for flashoagents.

Wraps an AppWorld world instance as a single code-execution Tool and builds a
ToolCallingAgent around it, so the full automem-runtime-v1 memory lifecycle
(BEGIN guidance injection, one summary-boundary refresh, trajectory capture)
is inherited from ToolCallingAgent unchanged.
"""

from __future__ import annotations

from typing import Any

from flashoagents.agents import ToolCallingAgent
from flashoagents.base_agent import BaseAgent
from flashoagents.tools import Tool

MAX_TOOL_OUTPUT_CHARS = 6000


class AppWorldExecuteTool(Tool):
    """Executes Python code inside an AppWorld task's live REPL."""

    name = "appworld_execute"
    description = (
        "Execute Python code in the AppWorld environment for the current task. "
        "A pre-loaded `apis` object exposes all app APIs (e.g. "
        "apis.spotify.show_playlist_library(access_token=...)); documentation is "
        "available via apis.api_docs.show_api_doc(app_name=..., api_name=...). "
        "Log in with the supervisor's stored passwords "
        "(apis.supervisor.show_profile() / show_account_passwords()) before "
        "calling app APIs. The REPL is stateful: variables persist across "
        "calls. Only printed output is returned. Finish with "
        "apis.supervisor.complete_task(status='success', answer=...)."
    )
    inputs = {
        "code": {
            "type": "string",
            "description": "Python source code to execute in the AppWorld REPL.",
        }
    }
    output_type = "string"

    def __init__(self, world: Any, max_output_chars: int = MAX_TOOL_OUTPUT_CHARS):
        super().__init__()
        self.world = world
        self.max_output_chars = max_output_chars
        self.is_initialized = True

    def forward(self, code: str) -> str:
        output = self.world.execute(code)
        text = "" if output is None else str(output)
        if len(text) > self.max_output_chars:
            text = (
                text[: self.max_output_chars]
                + f"\n...[truncated {len(text) - self.max_output_chars} chars]"
            )
        return text if text.strip() else "(no output; use print() to inspect values)"


class AppWorldAgent(BaseAgent):
    """Tool-calling agent operating inside one AppWorld task world."""

    def __init__(
        self,
        model,
        world: Any,
        summary_interval: int = 8,
        prompts_type: str = "appworld",
        max_steps: int = 40,
        memory_provider=None,
        **kwargs,
    ):
        super().__init__(model)
        execute_tool = AppWorldExecuteTool(world)
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=[execute_tool],
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider,
            **kwargs,
        )
