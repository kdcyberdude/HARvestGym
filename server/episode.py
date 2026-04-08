"""Episode data structures for HARvestGym."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurlCall:
    method: str
    url: str
    path: str          # normalized (IDs replaced with {id})
    headers: dict
    body: dict | str | None
    status_code: int
    response_body: Any
    response_headers: dict = field(default_factory=dict)


@dataclass
class Step:
    step_num: int
    tool: str          # browser_agent | search_endpoints | curl_exec | search_episode_data | done
    action: str        # raw tool call string
    result: Any        # tool return value
    curl_parsed: CurlCall | None = None


@dataclass
class Task:
    template_id: int                 # 1-7
    description: str                 # instantiated task string
    params: dict                     # e.g. {"product_name": "Radiant Tee", "sku": "MH01"}
    app: str                         # shopping | forum | wikipedia | shopping_admin
    base_url: str
    difficulty: str                  # easy | medium | hard


@dataclass
class Episode:
    task: Task
    steps: list[Step] = field(default_factory=list)
    session_state: dict = field(default_factory=dict)
    total_steps: int = 0
    terminated_by: str = ""          # "done_call" | "max_steps"


@dataclass
class EpisodeResult:
    task_score: float                # 0.0-1.0 from grader
    parameter_sourcing_score: float  # 0.0-1.0 from trajectory analysis
    auth_obtained: bool
    reward: float                    # final composite reward
    details: dict = field(default_factory=dict)
