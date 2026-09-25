"""SurgScope: question-conditioned windowing for hour-long surgical video question answering."""
from surgscope.policy import branch, pick_grid
from surgscope.prompts import system_prompt, user_content
from surgscope.routing import route

__version__ = "1.0.0"
__all__ = ["SurgScope", "route", "branch", "pick_grid", "user_content", "system_prompt"]


def __getattr__(name):
    if name == "SurgScope":          # imported lazily: it pulls in torch and ms-swift
        from surgscope.pipeline import SurgScope
        return SurgScope
    raise AttributeError(name)
