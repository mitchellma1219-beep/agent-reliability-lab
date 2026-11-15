"""Agent protocol, adapters and the bundled simulated agents."""

from __future__ import annotations

from .adapters import (
    ChatModel,
    FunctionAgent,
    Message,
    ModelTurn,
    ScriptedAgent,
    ScriptedCall,
    ToolLoopAgent,
    ToolRequest,
)
from .base import Agent, AgentContext, AgentResponse, ToolBox, infer_status
from .simulated import (
    FAILURE_MODES,
    Behaviour,
    SimulatedAgent,
    faulty_agent,
    reference_agent,
)

__all__ = [
    "FAILURE_MODES",
    "Agent",
    "AgentContext",
    "AgentResponse",
    "Behaviour",
    "ChatModel",
    "FunctionAgent",
    "Message",
    "ModelTurn",
    "ScriptedAgent",
    "ScriptedCall",
    "SimulatedAgent",
    "ToolBox",
    "ToolLoopAgent",
    "ToolRequest",
    "faulty_agent",
    "infer_status",
    "reference_agent",
]
