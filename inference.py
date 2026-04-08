"""
HARvestGym — Inference Script
==============================

Runs the RL agent (driven by an LLM via OpenAI client) through three tasks:
  1. har_classify_easy   — Template 1: list products in a category
  2. har_classify_medium — Template 3: add product to guest cart
  3. har_pipeline_hard   — Template 6: complete guest checkout

STDOUT FORMAT (strictly enforced by hackathon):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Provider selection (auto-detected from env vars):
  OpenRouter:  OPENROUTER_API_KEY=sk-or-xxx [MODEL_NAME=google/gemma-3-27b-it]
  HuggingFace: HF_TOKEN=hf_xxx              [MODEL_NAME=Qwen/Qwen2.5-72B-Instruct]

Usage:
  # OpenRouter (testing)
  OPENROUTER_API_KEY=sk-or-xxx uv run inference.py
  OPENROUTER_API_KEY=sk-or-xxx MODEL_NAME=google/gemma-3-27b-it uv run inference.py

  # HuggingFace (final submission)
  HF_TOKEN=hf_xxx uv run inference.py
  HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct uv run inference.py
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — auto-detect provider from env vars
# ---------------------------------------------------------------------------

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
_HF_TOKEN = os.getenv("HF_TOKEN")

if _OPENROUTER_KEY:
    # OpenRouter mode — great for testing with powerful models cheaply
    API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    API_KEY = _OPENROUTER_KEY
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31b-it")
    HF_TOKEN = _HF_TOKEN  # still needed for the env server itself
    print(f"[INFO] Provider: OpenRouter | Model: {MODEL_NAME}", flush=True)
elif _HF_TOKEN:
    # HuggingFace Inference Router — final submission target
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    API_KEY = _HF_TOKEN
    HF_TOKEN = _HF_TOKEN
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    print(f"[INFO] Provider: HuggingFace | Model: {MODEL_NAME}", flush=True)
else:
    raise ValueError(
        "No API key found. Set either:\n"
        "  OPENROUTER_API_KEY=sk-or-xxx   (for OpenRouter testing)\n"
        "  HF_TOKEN=hf_xxx                (for HuggingFace submission)"
    )

# ---------------------------------------------------------------------------
# Tool definitions — proper OpenAI function-calling format.
#
# Using the `tools` API (not response_format json_schema) is the correct way:
#  - Each tool has a name, description, and typed parameter schema
#  - The model sees exactly what each tool does and what args it needs
#  - tool_choice="required" forces the model to always call a tool (no free text)
#  - Works on OpenRouter, HuggingFace Router, and any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_agent",
            "description": (
                "Discovers all available API endpoints for the target web application "
                "by replaying recorded HTTP traffic (HAR files) and augmenting with a "
                "ground-truth API catalog. Returns a structured index of endpoints with "
                "methods, paths, and parameter schemas. "
                "MUST be called exactly once at step 1 before any other tool. "
                "Do NOT call again after step 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The natural language task description (e.g. 'Add Radiant Tee to cart')",
                    },
                    "url": {
                        "type": "string",
                        "description": "Base URL of the target application (e.g. 'http://host:7770/')",
                    },
                },
                "required": ["task", "url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_endpoints",
            "description": (
                "Search the discovered API endpoint catalog using a natural language query. "
                "Returns matching endpoint schemas including HTTP method, full path, "
                "required/optional parameters, authentication requirements, and example payloads. "
                "Use this after browser_agent to find the exact endpoint and payload structure "
                "before making a curl_exec call. "
                "Examples: 'create guest cart', 'add item to cart', 'set shipping address', "
                "'place order', 'get products by category'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the API operation you need (e.g. 'create guest cart', 'add item to cart')",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curl_exec",
            "description": (
                "Execute an HTTP request against the live application. "
                "Returns {status_code, headers, body} with the full API response. "
                "Session cookies and auth tokens are automatically injected — do NOT "
                "manually set Cookie or Authorization headers. "
                "Use proper curl syntax with -s (silent) flag. "
                "Always include -H 'Content-Type: application/json' for POST/PUT requests. "
                "Read the response body carefully — it contains IDs (cart_id, item_id, order_id) "
                "needed for subsequent steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Full curl command string. Examples:\n"
                            "  GET:  curl -s -X GET 'http://host/rest/V1/categories'\n"
                            "  POST: curl -s -X POST 'http://host/rest/V1/guest-carts' -H 'Content-Type: application/json'\n"
                            "  POST with body: curl -s -X POST 'http://host/rest/V1/guest-carts/CART_ID/items' "
                            "-H 'Content-Type: application/json' "
                            "-d '{\"cartItem\":{\"sku\":\"MH01-XS-Black\",\"qty\":1,\"quote_id\":\"CART_ID\"}}'"
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_episode_data",
            "description": (
                "Search all prior API responses collected during this episode for a specific value. "
                "Use when a previous curl_exec response was long/truncated and you need to find "
                "a specific item, ID, SKU, or field value from it. "
                "Examples: 'cart id from guest-carts response', 'product SKU for Radiant Tee', "
                "'category id for Gear'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What value you are looking for in the episode's response history (e.g. 'cart id', 'SKU for Radiant Tee')",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that the task is fully complete. Call this ONLY after you have "
                "successfully executed all required API calls and verified the outcome "
                "(e.g. item was added to cart, order was placed). "
                "Do NOT call done() as a fallback or when uncertain — it triggers final scoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Optional summary of what was accomplished (e.g. 'Added Radiant Tee to cart CART_ID, item_id=42')",
                    },
                },
                "additionalProperties": False,
            },
            "strict": False,  # result is optional
        },
    },
]

BENCHMARK = "harvgym"
MAX_STEPS = 20
TEMPERATURE = 0.2        # Lower temp → more deterministic tool calls
MAX_TOKENS = 1024        # More room for reasoning + JSON
SUCCESS_SCORE_THRESHOLD = 0.5

# Task definitions: use FIXED task descriptions so the model always knows
# exactly what to do (env.reset() may randomize, but we tell it the target)
TASKS = [
    {
        "task_name": "har_classify_easy",
        "template_id": 1,
        "description": "List products in the 'Gear' category",
        "app_base_url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/",
        "difficulty": "easy",
    },
    {
        "task_name": "har_classify_medium",
        "template_id": 3,
        "description": "Add 'Radiant Tee' (SKU: MH01-XS-Black) to a guest cart",
        "app_base_url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/",
        "difficulty": "medium",
    },
    {
        "task_name": "har_pipeline_hard",
        "template_id": 6,
        "description": "Complete a full guest checkout for 'Radiant Tee' (SKU: MH01-XS-Black)",
        "app_base_url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/",
        "difficulty": "hard",
    },
]

# ---------------------------------------------------------------------------
# Logging helpers (hackathon format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt — lean, since tool descriptions carry the full detail
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an API agent completing real-world tasks on a live Magento e-commerce application
by calling HTTP APIs in the correct sequence.

WORKFLOW:
1. Call browser_agent (step 1 only) to discover all available API endpoints.
2. Call search_endpoints to find the exact endpoint schema you need.
3. Call curl_exec to execute the HTTP request. Read the response — it contains IDs for next steps.
4. Repeat steps 2-3 for each action in the task (create cart → add item → set address → place order).
5. Call done() only after the task is fully accomplished.

KEY FACTS about Magento REST API (http://host:7770/rest/V1/):
- Guest cart flow: POST /guest-carts → returns cartId string
- Add item: POST /guest-carts/{cartId}/items  body: {"cartItem":{"sku":"...","qty":1,"quote_id":"{cartId}"}}
- Shipping: POST /guest-carts/{cartId}/shipping-information
- Place order: PUT /guest-carts/{cartId}/order

RULES:
- Call browser_agent exactly once at step 1.
- Always call search_endpoints before curl_exec to get the correct path and payload.
- Cart IDs, item IDs, and order IDs come from curl_exec responses — read them carefully.
- Do not call done() until the task is verified complete.
""").strip()


# ---------------------------------------------------------------------------
# LLM agent loop
# ---------------------------------------------------------------------------

def _format_result_for_context(result: Any, max_chars: int = 3000) -> str:
    """Format tool result for the LLM context — more generous truncation."""
    if result is None:
        return "null"
    try:
        text = json.dumps(result, indent=2)
    except Exception:
        text = str(result)

    if len(text) <= max_chars:
        return text

    # Smart truncation: keep beginning (has structure/IDs) and hint at truncation
    kept = text[:max_chars]
    # Try to close the JSON gracefully at the last complete line
    last_newline = kept.rfind("\n")
    if last_newline > max_chars * 0.8:
        kept = kept[:last_newline]
    return kept + f"\n... [truncated, {len(text) - max_chars} chars omitted — use search_episode_data to find specific values]"


def build_user_prompt(task_desc: str, app_base_url: str, step: int,
                      last_result: Any, history: List[dict],
                      session_state: dict) -> str:
    """Build the user prompt for each step."""
    history_lines = []
    if history:
        # Show last 8 steps with meaningful result summaries
        for h in history[-8:]:
            result = h.get("result", {})
            # For curl results: show status_code + first 200 chars of body
            if isinstance(result, dict) and "status_code" in result:
                body_preview = str(result.get("body", ""))[:300]
                result_summary = f'status={result["status_code"]} body={body_preview}'
            else:
                result_summary = str(result)[:300]
            history_lines.append(
                f"  Step {h['step']}: {h['tool']}({json.dumps(h.get('args', {}))[:100]}) "
                f"→ {result_summary}"
            )

    session_str = json.dumps(session_state, indent=2)[:500] if session_state else "{}"
    last_result_str = _format_result_for_context(last_result)

    return textwrap.dedent(f"""
    TASK: {task_desc}
    APP URL: {app_base_url}
    STEP: {step}/{MAX_STEPS}

    SESSION STATE (cookies/tokens auto-managed):
    {session_str}

    LAST TOOL RESULT:
    {last_result_str}

    HISTORY (last {len(history_lines)} steps):
    {chr(10).join(history_lines) if history_lines else "  (none yet)"}

    What is your next tool call? Output ONLY the JSON object.
    """).strip()


def get_model_action(client: OpenAI, task_desc: str, app_base_url: str,
                     step: int, last_result: Any, history: List[dict],
                     session_state: dict) -> dict:
    """Ask the LLM for the next action. Returns parsed tool call dict."""
    user_prompt = build_user_prompt(task_desc, app_base_url, step,
                                    last_result, history, session_state)

    # OpenRouter needs attribution headers; harmless on other providers
    extra_headers = {}
    if _OPENROUTER_KEY:
        extra_headers = {
            "HTTP-Referer": "https://huggingface.co/spaces/kdcyberdude/HARvestGym",
            "X-Title": "HARvestGym",
        }

    try:
        # Use the OpenAI tools API — each tool has name + description + typed params.
        # tool_choice="required" forces the model to always call a tool (no free text).
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=TOOLS,
            tool_choice="required",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            extra_headers=extra_headers if extra_headers else None,
        )

        choice = completion.choices[0]
        # Native tool call response (preferred — gives us structured args directly)
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"[DEBUG] Tool call: {tool_name}({list(args.keys())})", flush=True)
            return {"tool": tool_name, "args": args}

        # Some providers return plain text even with tools (fallback)
        text = (choice.message.content or "").strip()
        print(f"[DEBUG] No tool_calls in response, trying text parse: {text[:100]}", flush=True)
        return _parse_text_fallback(text, step, task_desc, app_base_url)

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        if step == 1:
            return {"tool": "browser_agent", "args": {"task": task_desc, "url": app_base_url}}
        return {"tool": "done", "args": {"result": f"LLM error: {exc}"}}


def _parse_text_fallback(text: str, step: int, task_desc: str, app_base_url: str) -> dict:
    """Last-resort parser if the model returns plain text instead of a tool call."""
    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    # Extract first JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    print(f"[DEBUG] Text fallback failed: {text[:200]}", flush=True)
    if step == 1:
        return {"tool": "browser_agent", "args": {"task": task_desc, "url": app_base_url}}
    if "done" in text.lower():
        return {"tool": "done", "args": {}}
    return {"tool": "done", "args": {"result": f"Parse error: {text[:100]}"}}


# ---------------------------------------------------------------------------
# Single task episode runner
# ---------------------------------------------------------------------------

async def run_episode(task_config: dict, client: OpenAI) -> dict:
    """
    Run a single episode for one task.
    Returns: {"task_name", "success", "steps", "score", "rewards"}
    """
    from server.models import HARvestGymEnvironment, HarvestGymAction

    task_name = task_config["task_name"]
    template_id = task_config["template_id"]
    task_description = task_config["description"]
    app_base_url = task_config["app_base_url"]

    # Pin the template via env var so reset() samples from the right pool
    os.environ["HARVGYM_TASK"] = task_name   # use name, not int

    env = HARvestGymEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_result = None
    history: List[dict] = []
    session_state: dict = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        # CRITICAL: use the env-sampled task description — the judge grades exactly
        # what env.reset() returned (random category/product), not our hardcoded string.
        task_desc = obs.task or task_description
        base_url = obs.app_base_url or app_base_url

        for step in range(1, MAX_STEPS + 1):
            if getattr(obs, "done", False):
                break

            action_dict = get_model_action(
                client=client,
                task_desc=task_desc,
                app_base_url=base_url,
                step=step,
                last_result=last_result,
                history=history,
                session_state=session_state,
            )

            tool = action_dict.get("tool", "done")
            args = action_dict.get("args", {})

            action_str = f"{tool}({json.dumps(args)[:150]})"
            error_str = None

            try:
                action = HarvestGymAction(tool=tool, args=args)
                obs = env.step(action)

                reward = float(obs.reward or 0.0)
                done = bool(obs.done)
                last_result = obs.last_tool_result
                session_state = dict(obs.session_state or {})

                history.append({
                    "step": step,
                    "tool": tool,
                    "args": args,
                    "result": last_result,
                })

            except Exception as exc:
                reward = -0.1
                done = False
                error_str = str(exc)[:200]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        # Score: terminal reward from judge dominates.
        # Reward range by design: terminal success = +2 to +5, terminal fail = -1.5
        # Use a generous baseline so partial credit shows up.
        total_reward = sum(rewards)
        # Normalise to [0,1]: shift by +1.5 (min), divide by max-possible per task
        # Template 1 max=2, Template 3 max=3.5, Template 6 max=5 → use 5.0 as ceiling
        score = max(0.0, min(1.0, (total_reward + 1.5) / (5.0 + 1.5)))
        success = total_reward >= 0.5   # any positive terminal reward = success

    except Exception as exc:
        error_str = str(exc)[:200]
        print(f"[DEBUG] Episode error: {error_str}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_name": task_name,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_config in TASKS:
        result = await run_episode(task_config, client)
        results.append(result)

    # Summary
    print("\n[SUMMARY]", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  [{status}] {r['task_name']} — score={r['score']:.2f} steps={r['steps']}",
            flush=True,
        )

    overall_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  overall_score={overall_score:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
