"""HARvestGym client for interacting with the environment server."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .server.models import HarvestGymAction, HarvestGymObservation
except ModuleNotFoundError:
    from server.models import HarvestGymAction, HarvestGymObservation


class HARvestGymEnv(EnvClient[HarvestGymAction, HarvestGymObservation, State]):
    """
    Client for the HARvestGym Environment.

    Example:
        >>> async with HARvestGymEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     result = await env.step(HarvestGymAction(
        ...         tool="browser_agent",
        ...         args={"task": "List products in category Gear",
        ...               "url": "http://ec2-.../"}
        ...     ))
    """

    def _step_payload(self, action: HarvestGymAction) -> Dict:
        return {
            "tool": action.tool,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HarvestGymObservation]:
        obs_data = payload.get("observation", {})
        observation = HarvestGymObservation(
            task=obs_data.get("task", ""),
            app_base_url=obs_data.get("app_base_url", ""),
            last_tool_result=obs_data.get("last_tool_result"),
            history=obs_data.get("history", []),
            session_state=obs_data.get("session_state", {}),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
