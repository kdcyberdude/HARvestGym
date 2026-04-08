"""
HARvestGym Environment — core OpenEnv models.py

Implements the OpenEnv spec:
  - Observation, Action, Reward as Pydantic models
  - reset() → initial observation + clean state
  - step(action) → (observation, reward, done, info)
  - state() → current state snapshot

The environment manages episode state, dispatches tool calls, computes per-step
rewards, and invokes the judge at episode end.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from pydantic import Field

from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HarvestGymObservation(BaseObservation):
    """What the RL agent sees at each step."""

    task: str = Field(default="", description="Natural language task description")
    app_base_url: str = Field(default="", description="Root URL of the target application")
    last_tool_result: Any = Field(default=None, description="Result of last tool call")
    history: list[dict] = Field(default_factory=list, description="Full episode trajectory")
    session_state: dict = Field(default_factory=dict, description="Auto-managed cookies/tokens")
    step_count: int = Field(default=0)
    max_steps: int = Field(default=20)
    available_tools: list[str] = Field(
        default_factory=lambda: [
            "browser_agent(task, url) — discover API endpoints from HAR + catalog; call once at step 1",
            "search_endpoints(query) — find endpoint schema by natural language query",
            "curl_exec(command) — execute HTTP request; returns {status_code, body}",
            "search_episode_data(query) — search prior responses for a specific value/ID",
            "done(result?) — signal task complete; triggers final scoring",
        ]
    )


class HarvestGymAction(BaseAction):
    """One tool call from the RL agent."""

    tool: str = Field(..., description="Tool name: browser_agent|search_endpoints|curl_exec|search_episode_data|done")
    args: dict = Field(default_factory=dict, description="Tool-specific arguments")


class HarvestGymReward(BaseObservation):
    """Reward signal (returned as part of the observation)."""

    value: float = Field(default=0.0, description="Scalar reward for this step")
    breakdown: dict = Field(default_factory=dict, description="Per-signal reward components")


# ---------------------------------------------------------------------------
# Per-step reward constants
# ---------------------------------------------------------------------------

REWARD_VALID_API_CALL = 0.2      # curl_exec returns 2xx
REWARD_NEW_PATH = 0.1            # curl path not seen before this episode
REWARD_CORRECT_PARAM = 0.25      # judge: correct parameter sourcing (applied at end)
REWARD_SESSION_VALUE = 0.1       # auth token/cookie correctly used
PENALTY_REPEATED_CALL = -0.15    # exact duplicate curl command
PENALTY_BROWSER_AGENT_AGAIN = -0.3  # browser_agent called after step 1
PENALTY_MALFORMED_CURL = -0.1    # curl can't be parsed/executed
PENALTY_4XX = -0.05              # recoverable HTTP error

MAX_STEPS = 20

# ---------------------------------------------------------------------------
# Task templates
# ---------------------------------------------------------------------------

TEMPLATE_META = {
    1: {"tier": "easy",   "app": "shopping",        "base_url_port": 7770},
    2: {"tier": "easy",   "app": "wikipedia",       "base_url_port": 8888},
    3: {"tier": "medium", "app": "shopping",        "base_url_port": 7770},
    4: {"tier": "medium", "app": "forum",           "base_url_port": 9999},
    5: {"tier": "hard",   "app": "forum",           "base_url_port": 9999},
    6: {"tier": "hard",   "app": "shopping",        "base_url_port": 7770},
    7: {"tier": "hard",   "app": "shopping_admin",  "base_url_port": 7780},
}

EC2_HOST = os.environ.get("EC2_HOST", "ec2-16-59-2-56.us-east-2.compute.amazonaws.com")

TASK_NAME_TO_TEMPLATE = {
    "har_classify_easy": 1,
    "har_classify_medium": 3,
    "har_pipeline_hard": 6,
}

TEMPLATE_DESCRIPTIONS = {
    1: "List products in category {category_name}",
    2: "Retrieve the Wikipedia article for '{title}'",
    3: "Add '{product_name}' to a guest cart",
    4: "Retrieve all posts in the '{forum_category}' forum (you must log in first)",
    5: "Create a forum post titled '{title}' in the '{category}' forum",
    6: "Complete a guest checkout for '{product_name}'",
    7: "Create a new product in the admin panel with SKU '{sku}' and price {price}",
}


def _load_parameter_pools() -> dict:
    pools_path = Path(__file__).parent.parent / "parameter_pools.json"
    if pools_path.exists():
        with open(pools_path) as f:
            return json.load(f)
    return {}


def _sample_task(template_id: int, parameter_pools: dict) -> tuple[str, dict, str]:
    """
    Sample a task instance from the parameter pool.

    Returns: (task_description, params_dict, app_base_url)
    """
    meta = TEMPLATE_META[template_id]
    pool_key = f"template_{template_id}"
    pool_data = parameter_pools.get(pool_key, {})
    pool = pool_data.get("pool", {})

    params: dict = {}

    if template_id == 1:
        items = pool.get("category_name", [{"name": "Gear", "category_id": 3}])
        chosen = random.choice(items)
        params = {"category_name": chosen["name"], "category_id": chosen.get("category_id")}
        description = TEMPLATE_DESCRIPTIONS[1].format(**params)

    elif template_id == 2:
        items = pool.get("title", [{"title": "Python (programming language)", "expected_slug": "Python_(programming_language)"}])
        if not items:
            items = [{"title": "Python (programming language)", "expected_slug": "Python_(programming_language)"}]
        chosen = random.choice(items)
        title = chosen.get("title", chosen) if isinstance(chosen, dict) else chosen
        params = {"title": title, "expected_slug": chosen.get("expected_slug", title.replace(" ", "_"))}
        description = TEMPLATE_DESCRIPTIONS[2].format(**params)

    elif template_id == 3:
        items = pool.get("product_name", [{"name": "Radiant Tee", "sku": "MH01"}])
        if not items:
            items = [{"name": "Radiant Tee", "sku": "MH01"}]
        chosen = random.choice(items)
        product_name = chosen.get("name", chosen) if isinstance(chosen, dict) else chosen
        sku = chosen.get("sku", "") if isinstance(chosen, dict) else ""
        params = {"product_name": product_name, "sku": sku}
        description = TEMPLATE_DESCRIPTIONS[3].format(**params)

    elif template_id == 4:
        items = pool.get("forum_category", [{"slug": "general", "name": "General"}])
        if not items:
            items = [{"slug": "general", "name": "General"}]
        chosen = random.choice(items)
        forum_cat = chosen.get("slug", chosen.get("name", "general")) if isinstance(chosen, dict) else chosen
        params = {"forum_category": forum_cat}
        description = TEMPLATE_DESCRIPTIONS[4].format(**params)

    elif template_id == 5:
        categories = pool.get("forum_category", [{"slug": "general"}])
        titles = pool.get("post_title", ["Testing the API agent framework"])
        if not categories:
            categories = [{"slug": "general"}]
        if not titles:
            titles = ["Testing the API agent framework"]
        chosen_cat = random.choice(categories)
        chosen_title = random.choice(titles) if isinstance(titles[0], str) else random.choice(titles).get("title", "Test post")
        forum_cat = chosen_cat.get("slug", "general") if isinstance(chosen_cat, dict) else chosen_cat
        params = {"title": chosen_title, "category": forum_cat}
        description = TEMPLATE_DESCRIPTIONS[5].format(**params)

    elif template_id == 6:
        items = pool.get("product_name", [{"name": "Radiant Tee", "sku": "MH01"}])
        if not items:
            items = [{"name": "Radiant Tee", "sku": "MH01"}]
        chosen = random.choice(items)
        product_name = chosen.get("name", chosen) if isinstance(chosen, dict) else chosen
        sku = chosen.get("sku", "") if isinstance(chosen, dict) else ""
        params = {"product_name": product_name, "sku": sku}
        description = TEMPLATE_DESCRIPTIONS[6].format(**params)

    elif template_id == 7:
        items = pool.get("admin_sku", [{"sku": "HAR-TEST-001", "price": "29.99"}])
        if not items:
            items = [{"sku": "HAR-TEST-001", "price": "29.99"}]
        chosen = random.choice(items)
        sku = chosen.get("sku", "HAR-TEST-001") if isinstance(chosen, dict) else chosen
        price = str(chosen.get("price", "29.99")) if isinstance(chosen, dict) else "29.99"
        params = {"sku": sku, "price": price}
        description = TEMPLATE_DESCRIPTIONS[7].format(**params)

    else:
        params = {}
        description = f"Template {template_id}"

    port = meta["base_url_port"]
    base_url = f"http://{EC2_HOST}:{port}/"
    return description, params, base_url


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HARvestGymEnvironment(Environment):
    """
    HARvestGym: RL environment for training API-native web agents.

    The agent must discover and execute the correct sequence of HTTP API calls
    to complete real-world tasks on live web applications — starting from only
    a task description and a URL, with no prior knowledge of the API schema.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._parameter_pools = _load_parameter_pools()
        self._current_task = None        # Task dataclass
        self._episode = None             # Episode dataclass
        self._session_state: dict = {}
        self._episode_store: dict = {}   # embeddings, BM25 corpus, etc.
        self._called_paths: set = set()  # for new-path reward
        self._last_curl_commands: list = []  # for duplicate detection
        self._step_rewards: list[float] = []
        self._done = False

        # Determine default template from env var
        self._task_name = os.environ.get("HARVGYM_TASK", "har_classify_easy")

    # -----------------------------------------------------------------------
    # Metadata — exposed via GET /metadata and used by RL training loops
    # to build the system prompt and tool definitions automatically.
    # -----------------------------------------------------------------------

    def get_metadata(self):  # → EnvironmentMetadata
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="HARvestGym",
            version="1.0.0",
            author="kdcyberdude",
            description=(
                "HARvestGym is a real-world RL environment for training API-native web agents. "
                "The agent receives a natural language task (e.g. 'Add Radiant Tee to cart') "
                "and must discover and execute the correct sequence of HTTP API calls on a live "
                "Magento e-commerce application — starting from only a URL and a task description. "
                "\n\n"
                "TOOLS (5 available):\n"
                "1. browser_agent(task, url)\n"
                "   Discovers all API endpoints from recorded HAR traffic + ground-truth catalog. "
                "   Returns structured endpoint index. Call EXACTLY ONCE at step 1.\n\n"
                "2. search_endpoints(query)\n"
                "   Semantic/keyword search over the discovered endpoint catalog. "
                "   Returns method, path, parameter schema, and auth requirements. "
                "   Use before each curl_exec to find the correct endpoint and payload.\n\n"
                "3. curl_exec(command)\n"
                "   Execute a real HTTP request. Returns {status_code, headers, body}. "
                "   Session cookies/tokens are auto-injected. "
                "   Response body contains IDs (cartId, item_id, orderId) for subsequent steps.\n\n"
                "4. search_episode_data(query)\n"
                "   Search all prior API responses from this episode for a specific value. "
                "   Use when earlier responses were truncated and you need to retrieve an ID or field.\n\n"
                "5. done(result?)\n"
                "   Signal task completion. Triggers the deterministic judge and final scoring. "
                "   Call ONLY after all required API calls are verified successful.\n\n"
                "REWARD SIGNALS:\n"
                "+0.20  valid API call (2xx response)\n"
                "+0.10  new unique API path explored\n"
                "+0.25  correct parameter sourcing (judge)\n"
                "+0.10  auth token/cookie correctly propagated\n"
                "-0.05  4xx HTTP error\n"
                "-0.10  malformed curl command\n"
                "-0.15  exact duplicate curl call\n"
                "-0.30  browser_agent called after step 1\n"
                "-1.50  episode timeout (max 20 steps)\n"
                "\n"
                "TASK TIERS: easy (list products), medium (add to cart), hard (full checkout pipeline)"
            ),
            documentation_url="https://huggingface.co/spaces/kdcyberdude/HARvestGym",
        )

    def _get_template_id(self) -> int:
        """Resolve task name or template ID from env var."""
        task_name = self._task_name
        if task_name in TASK_NAME_TO_TEMPLATE:
            return TASK_NAME_TO_TEMPLATE[task_name]
        # Try integer
        try:
            tid = int(task_name)
            if 1 <= tid <= 7:
                return tid
        except (ValueError, TypeError):
            pass
        return 1  # default: easy

    def reset(self) -> HarvestGymObservation:
        """Reset environment: clear episode state, sample new task."""
        from .episode import Episode, Task

        template_id = self._get_template_id()
        description, params, base_url = _sample_task(template_id, self._parameter_pools)

        meta = TEMPLATE_META[template_id]
        self._current_task = Task(
            template_id=template_id,
            description=description,
            params=params,
            app=meta["app"],
            base_url=base_url,
            difficulty=meta["tier"],
        )

        self._episode = Episode(task=self._current_task)
        self._session_state = {}
        self._episode_store = {}
        self._called_paths = set()
        self._last_curl_commands = []
        self._step_rewards = []
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return HarvestGymObservation(
            task=description,
            app_base_url=base_url,
            last_tool_result=None,
            history=[],
            session_state={},
            step_count=0,
            max_steps=MAX_STEPS,
            done=False,
            reward=0.0,
            metadata={
                "template_id": template_id,
                "difficulty": meta["tier"],
                "app": meta["app"],
            },
        )

    def step(self, action: HarvestGymAction) -> HarvestGymObservation:  # type: ignore[override]
        """Execute one tool call and return the next observation."""
        from .episode import Step, CurlCall

        if self._done:
            # Episode already finished
            return self._make_obs(
                last_tool_result={"error": "Episode already done. Call reset()."},
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1
        step_num = self._state.step_count

        tool = action.tool.lower().strip()
        args = action.args or {}

        # Dispatch tool
        result, step_reward, done = self._dispatch_tool(tool, args, step_num)

        # Record step in episode
        step_obj = Step(
            step_num=step_num,
            tool=tool,
            action=f"{tool}({json.dumps(args)})",
            result=result,
        )

        # If curl_exec, parse the curl call for judge
        if tool == "curl_exec":
            command = args.get("command", "")
            try:
                from .tools.curl_exec import parse_curl_command
                parsed = parse_curl_command(command)
                from urllib.parse import urlparse
                path = urlparse(parsed["url"]).path if parsed["url"] else ""
                from .tools.browser_agent import _normalise_path
                norm_path = _normalise_path(path)

                resp = result if isinstance(result, dict) else {}
                step_obj.curl_parsed = CurlCall(
                    method=parsed["method"],
                    url=parsed["url"] or "",
                    path=norm_path,
                    headers=parsed["headers"],
                    body=parsed["body"],
                    status_code=resp.get("status_code", 0),
                    response_body=resp.get("body"),
                    response_headers=resp.get("headers", {}),
                )
            except Exception:
                pass

        if self._episode:
            self._episode.steps.append(step_obj)
            self._episode.total_steps = step_num

        self._step_rewards.append(step_reward)

        # Check max steps
        if step_num >= MAX_STEPS and not done:
            done = True
            if self._episode:
                self._episode.terminated_by = "max_steps"
            # Invoke judge
            judge_reward = self._invoke_judge()
            step_reward += judge_reward

        if done and self._episode and not self._episode.terminated_by:
            self._episode.terminated_by = "done_call"

        self._done = done

        # Build history entry
        history_entry = {
            "step": step_num,
            "tool": tool,
            "args": args,
            "result": result,
            "reward": step_reward,
        }
        if self._episode:
            history_for_obs = [
                {"step": s.step_num, "tool": s.tool, "result": s.result}
                for s in self._episode.steps
            ]
        else:
            history_for_obs = [history_entry]

        return HarvestGymObservation(
            task=self._current_task.description if self._current_task else "",
            app_base_url=self._current_task.base_url if self._current_task else "",
            last_tool_result=result,
            history=history_for_obs,
            session_state=dict(self._session_state),
            step_count=step_num,
            max_steps=MAX_STEPS,
            done=done,
            reward=step_reward,
            metadata={
                "step": step_num,
                "tool": tool,
                "step_reward": step_reward,
            },
        )

    def _dispatch_tool(self, tool: str, args: dict, step_num: int) -> tuple[Any, float, bool]:
        """
        Dispatch to the correct tool. Returns (result, step_reward, done).
        """
        reward = 0.0
        done = False

        if tool == "browser_agent":
            task = args.get("task", self._current_task.description if self._current_task else "")
            url = args.get("url", self._current_task.base_url if self._current_task else "")

            # Penalty if called after step 1
            if step_num > 1:
                reward += PENALTY_BROWSER_AGENT_AGAIN

            from .tools.browser_agent import run_browser_agent
            result = run_browser_agent(task, url, episode_store=self._episode_store)

        elif tool == "search_endpoints":
            query = args.get("query", "")
            from .tools.search_endpoints import search_endpoints
            result = search_endpoints(query, self._episode_store)

        elif tool == "curl_exec":
            command = args.get("command", "")
            if not command:
                return {"error": "curl_exec requires 'command' argument"}, PENALTY_MALFORMED_CURL, False

            # Duplicate detection
            if command in self._last_curl_commands:
                reward += PENALTY_REPEATED_CALL
            self._last_curl_commands.append(command)

            from .tools.curl_exec import curl_exec
            result = curl_exec(
                command=command,
                session_state=self._session_state,
                episode_store=self._episode_store,
                app_base_url=self._current_task.base_url if self._current_task else "",
            )

            status = result.get("status_code", 0)
            if status == -1 or "error" in result:
                reward += PENALTY_MALFORMED_CURL
            elif 200 <= status < 300:
                reward += REWARD_VALID_API_CALL
                # New path bonus
                from urllib.parse import urlparse
                from .tools.browser_agent import _normalise_path
                try:
                    parsed_for_path = __import__("shlex").split(command)
                    for t in parsed_for_path:
                        if t.startswith("http"):
                            path = _normalise_path(urlparse(t.strip("'\"")).path)
                            if path and path not in self._called_paths:
                                self._called_paths.add(path)
                                reward += REWARD_NEW_PATH
                            break
                except Exception:
                    pass
            elif 400 <= status < 500:
                reward += PENALTY_4XX

        elif tool == "search_episode_data":
            query = args.get("query", "")
            from .tools.search_episode_data import search_episode_data
            result = search_episode_data(query, self._episode_store)

        elif tool == "done":
            result_str = args.get("result", "")
            result = {"status": "done", "result": result_str}
            done = True
            # Invoke judge for final reward
            judge_reward = self._invoke_judge()
            reward += judge_reward

        else:
            result = {"error": f"Unknown tool: {tool}. Available: browser_agent, search_endpoints, curl_exec, search_episode_data, done"}
            reward += PENALTY_MALFORMED_CURL

        return result, reward, done

    def _invoke_judge(self) -> float:
        """Run the judge on the completed episode and return terminal reward."""
        if self._episode is None or self._current_task is None:
            return -1.5
        try:
            from .judge import evaluate
            episode_result = evaluate(self._episode)
            return episode_result.reward
        except Exception as e:
            print(f"[HARvestGym] Judge error: {e}", flush=True)
            return -1.5

    def _make_obs(self, last_tool_result: Any, reward: float, done: bool) -> HarvestGymObservation:
        return HarvestGymObservation(
            task=self._current_task.description if self._current_task else "",
            app_base_url=self._current_task.base_url if self._current_task else "",
            last_tool_result=last_tool_result,
            history=[],
            session_state=dict(self._session_state),
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
