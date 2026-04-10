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
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Verbose mode — set VERBOSE=1 for detailed per-step debugging.
# Keep disabled (default) for hackathon submission to avoid stdout noise.
# ---------------------------------------------------------------------------

VERBOSE = os.getenv("VERBOSE", "0").strip() == "1"


def vprint(*args) -> None:
    """Print only when VERBOSE=1."""
    if VERBOSE:
        print(*args, flush=True)


def vdump(label: str, obj: Any, max_chars: int = 2000) -> None:
    """Pretty-print a labelled object when verbose."""
    if not VERBOSE:
        return
    try:
        text = json.dumps(obj, indent=2)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text)-max_chars} chars]"
    print(f"\n{'─'*60}\n[VERBOSE] {label}\n{'─'*60}\n{text}\n", flush=True)


# ---------------------------------------------------------------------------
# Configuration — auto-detect provider from env vars
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is required but not set.\n"
        "Usage: HF_TOKEN=hf_xxx uv run inference.py"
    )

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if _OPENROUTER_KEY:
    # OpenRouter mode — useful for local testing with alternative models
    API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    API_KEY = _OPENROUTER_KEY
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31b-it")
    print(f"[INFO] Provider: OpenRouter | Model: {MODEL_NAME}", flush=True)
else:
    # HuggingFace Inference Router — final submission target
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    API_KEY = HF_TOKEN
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
    print(f"[INFO] Provider: HuggingFace | Model: {MODEL_NAME}", flush=True)

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
                "Discovers API endpoints available on the target web application by "
                "replaying real browser traffic recorded in HAR files. Returns a "
                "structured index of observed endpoints with HTTP methods, paths, "
                "request/response schemas, and headers (including any auth headers seen). "
                "Call this ONCE at step 1 to build the endpoint index. Do not call again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task you need to accomplish (used to prioritise relevant endpoints)",
                    },
                    "url": {
                        "type": "string",
                        "description": "Base URL of the target application",
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
                "Semantic search over the endpoints and it's details found by the browser_agent. "
                "Returns matching endpoint schemas: HTTP method, full path, required parameters, "
                "authentication requirements (bearer token, cookie, etc.), and example payloads. "
                "Use this before every curl_exec call to confirm the correct endpoint shape. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the operation you need "
                                       "(e.g. 'authenticate user', 'list products in category', "
                                       "'add item to cart', 'place order')",
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
                "Execute an HTTP request against the live application and return the response. "
                "Response contains: status_code, headers, body. "
                "For HTML pages, body is a structured summary: page title, forms with action URLs "
                "and field values (product IDs, form_key, etc.), and visible text. "
                "IMPORTANT: When the body shows '[Forms — N found]' with POST actions containing "
                "'/checkout/cart/add/...', the 'product' field IS the product ID and the action "
                "URL IS the add-to-cart endpoint — use these directly without calling "
                "search_episode_data again. "
                "Session state (cookies, auth tokens) is automatically managed — previously "
                "obtained tokens are injected into subsequent requests automatically. "
                "If the response is truncated or you need a value from an earlier response, "
                "use search_episode_data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Full curl command string (use -s for silent mode). "
                            "Include -H 'Content-Type: application/json' for POST/PUT/PATCH. "
                            "Example: curl -s -X POST 'http://host/api/endpoint' "
                            "-H 'Content-Type: application/json' -d '{\"key\":\"value\"}'"
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
                "Semantic search over all API responses collected during this episode. "
                "Full response bodies are stored untruncated — this tool finds the right "
                "response and returns a compact preview with a note showing the total "
                "number of matching objects (e.g. '47 items total — showing first 3'). "
                "Use more specific queries to drill into a particular value. "
                "Examples: 'id for category Gear', 'SKU for Radiant Tee', "
                "'cart id', 'authentication token', 'order id after checkout'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you are looking for in the response history of the curl commands you executed "
                                       "(e.g. 'category id for Pants', 'cart id', 'token')",
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
                "Signal that the task is complete and trigger final scoring. "
                "Call this immediately after the response that fulfills the task objective. "
                "Do not make further API calls once the goal is met — call done() next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "additionalProperties": False,
            },
            "strict": False,
        },
    },
]

BENCHMARK = "harvgym"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 64000
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Task bank — 5 easy (T1: list products), 5 medium (T3: add to cart),
# 5 hard (T6: guest checkout).
#
# For hackathon submission only the first easy/medium/hard is run.
# Full evaluation runs all 15 sequentially to measure generalisation.
# ---------------------------------------------------------------------------
_SHOP = "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/"

def _load_parameter_pools_for_tasks() -> dict:
    pools_path = Path(__file__).with_name("parameter_pools.json")
    with open(pools_path) as f:
        return json.load(f)


_TASK_PARAMETER_POOLS = _load_parameter_pools_for_tasks()


def _lookup_category_params(category_name: str) -> dict:
    categories = _TASK_PARAMETER_POOLS.get("template_1", {}).get("pool", {}).get("category_name", [])
    for item in categories:
        if item.get("name") == category_name:
            return {
                "category_name": item["name"],
                "category_id": item.get("category_id"),
            }
    raise ValueError(f"Unknown category in parameter_pools.json: {category_name}")


def _lookup_product_params(product_name: str, template_id: int) -> dict:
    products = _TASK_PARAMETER_POOLS.get(f"template_{template_id}", {}).get("pool", {}).get("product_name", [])
    for item in products:
        if item.get("name") == product_name:
            return {
                "product_name": item["name"],
                "sku": item.get("sku", ""),
                "product_id": item.get("product_id"),
            }
    raise ValueError(
        f"Unknown product in parameter_pools.json for template {template_id}: {product_name}"
    )


def _make_easy_task(task_name: str, category_name: str) -> dict:
    return {
        "task_name": task_name,
        "template_id": 1,
        "difficulty": "easy",
        "description": f"List products in the '{category_name}' category",
        "app_base_url": _SHOP,
        "task_params": _lookup_category_params(category_name),
    }


def _make_product_task(task_name: str, template_id: int, difficulty: str,
                       description: str, product_name: str) -> dict:
    return {
        "task_name": task_name,
        "template_id": template_id,
        "difficulty": difficulty,
        "description": description,
        "app_base_url": _SHOP,
        "task_params": _lookup_product_params(product_name, template_id),
    }


TASKS_EASY = [
    _make_easy_task("easy_list_pants", "Pants"),
    _make_easy_task("easy_list_bags", "Bags"),
    _make_easy_task("easy_list_jackets", "Jackets"),
    _make_easy_task("easy_list_hoodies", "Hoodies"),
    _make_easy_task("easy_list_shoes", "Shoes"),
]

TASKS_MEDIUM = [
    _make_product_task(
        "medium_cart_camera_backpack",
        3,
        "medium",
        "Add 'Camera Backpack Bagsmar DSLR Waterproof' to a guest cart",
        "Camera Backpack Bagsmar DSLR Waterproof",
    ),
    _make_product_task(
        "medium_cart_flannel_jacket",
        3,
        "medium",
        "Add 'Noldares Flannel Jacket For Men Plaid' to a guest cart",
        "Noldares Flannel Jacket For Men Plaid",
    ),
    _make_product_task(
        "medium_cart_champion_hoodie",
        3,
        "medium",
        "Add 'Champion Hoodie Big And Tall Zip Up' to a guest cart",
        "Champion Hoodie Big And Tall Zip Up",
    ),
    _make_product_task(
        "medium_cart_cargo_pants",
        3,
        "medium",
        "Add 'Mens Slim Fit Cargo Pants Athletic' to a guest cart",
        "Mens Slim Fit Cargo Pants Athletic",
    ),
    _make_product_task(
        "medium_cart_leather_jacket",
        3,
        "medium",
        "Add 'Inesver Womens Leather Jacket Open Front' to a guest cart",
        "Inesver Womens Leather Jacket Open Front",
    ),
]

TASKS_HARD = [
    _make_product_task(
        "hard_checkout_ripstop_pants",
        6,
        "hard",
        "Complete a full guest checkout for 'Mens Ripstop Cargo Pants Tactical Hiking'",
        "Mens Ripstop Cargo Pants Tactical Hiking",
    ),
    _make_product_task(
        "hard_checkout_flannel_jacket",
        6,
        "hard",
        "Complete a full guest checkout for 'Noldares Flannel Jacket For Men Plaid'",
        "Noldares Flannel Jacket For Men Plaid",
    ),
    _make_product_task(
        "hard_checkout_champion_hoodie",
        6,
        "hard",
        "Complete a full guest checkout for 'Champion Hoodie Big And Tall Zip Up'",
        "Champion Hoodie Big And Tall Zip Up",
    ),
    _make_product_task(
        "hard_checkout_fleece_jacket",
        6,
        "hard",
        "Complete a full guest checkout for 'Womens Fleece Jacket With Hood Winter'",
        "Womens Fleece Jacket With Hood Winter",
    ),
    _make_product_task(
        "hard_checkout_totes_boots",
        6,
        "hard",
        "Complete a full guest checkout for 'Totes Womens Cold Weather Boots Nicole'",
        "Totes Womens Cold Weather Boots Nicole",
    ),
]

# Default: first of each tier (hackathon submission format)
TASKS = [ TASKS_EASY[0], TASKS_MEDIUM[0], TASKS_MEDIUM[1], TASKS_HARD[0]]

# Set EVAL_MODE=full to run all 1By default, we have three tasks.5; EVAL_MODE=easy/medium/hard to run only that tier
_EVAL_MODE = os.getenv("EVAL_MODE", "").strip().lower()
if _EVAL_MODE == "full":
    TASKS = TASKS_EASY + TASKS_MEDIUM + TASKS_HARD
elif _EVAL_MODE == "easy":
    TASKS = TASKS_EASY
elif _EVAL_MODE == "one":
    TASKS = [TASKS_MEDIUM[1]]
elif _EVAL_MODE == "medium":
    TASKS = TASKS_MEDIUM
elif _EVAL_MODE == "hard":
    TASKS = TASKS_HARD

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
You are an API agent. Your goal is to complete a real-world task on a live web application
by calling its HTTP APIs in the correct order using the tools provided.

WORKFLOW:
1. Call browser_agent once at step 1 to build an index of the application's endpoints.
2. Use search_endpoints before each API call to find the correct path, method, and required parameters.
3. Execute HTTP requests with curl_exec in the correct dependency order. Read every response
   carefully — IDs, tokens, and error messages in responses are required inputs for (or
   corrective signals for) subsequent calls.
4. If a prior response contains a value you need now, use search_episode_data to retrieve it.
5. Call done() as soon as the task objective is met.

PRINCIPLES:
- Always discover before you act: browser_agent first, then search_endpoints.
- Extract every ID, token, and key from API responses and use them in subsequent calls.
- If a request returns an auth error, find and call the auth endpoint first, then retry.
- Never fabricate IDs or values — they must come from actual API responses.
- Once the task is done, call done() immediately — do not make additional calls.
- Some tasks require a sequence of dependent API calls where the output of one call
  (an ID, token, or key) is the required input to the next. Identify these dependencies
  before acting: plan the call sequence, then execute step by step.
- Never call the same endpoint repeatedly hoping for a different result. If a call already
  succeeded, move on to the next step. Repeating the same call wastes steps and incurs a
  penalty.
- Do not brute-force or vary parameters at random. If a call fails, read the error message
  in LAST TOOL RESULT, diagnose the cause logically, and use that understanding to form the
  correct next request.
- If you are partway through a multi-step task and a required ID or token is missing, use
  search_episode_data to retrieve it from an earlier response before making a new call.
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
        for h in history:
            result = h.get("result", {})
            if isinstance(result, dict) and "status_code" in result:
                body_preview = str(result.get("body", ""))[:800]
                result_summary = f'status={result["status_code"]} body={body_preview}'
            else:
                result_summary = str(result)[:300]
            history_lines.append(
                f"  Step {h['step']}: {h['tool']}({json.dumps(h.get('args', {}))[:100]}) "
                f"→ {result_summary}"
            )

    session_str = json.dumps(session_state, indent=2)[:500] if session_state else "{}"
    last_result_str = _format_result_for_context(last_result)

    # Highlight form_key if available — it's needed for HTML form POSTs
    form_key_hint = ""
    if session_state.get("form_key"):
        form_key_hint = f"\nFORM_KEY (auto-extracted, use in POST body): {session_state['form_key']}"

    return textwrap.dedent(f"""
    TASK: {task_desc}
    APP URL: {app_base_url}
    STEP: {step}/{MAX_STEPS}

    SESSION STATE (cookies/tokens auto-managed):{form_key_hint}
    {session_str}

    LAST TOOL RESULT:
    {last_result_str}

    HISTORY (all {len(history_lines)} steps so far):
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

    vprint(f"\n{'═'*60}")
    vprint(f"[VERBOSE] === LLM CALL — step {step} ===")
    vdump("SYSTEM PROMPT", SYSTEM_PROMPT)
    vdump("USER PROMPT", user_prompt)

    # Retry loop — backs off on 429 rate limits, never calls done() on a transient error
    _MAX_RETRIES = 3
    _BASE_DELAY  = 3   # seconds before first retry
    for _attempt in range(_MAX_RETRIES):
        try:
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

            choice = completion.choices[0] if completion.choices else None

            vdump(f"RAW COMPLETION (step {step}, attempt {_attempt+1})", {
                "finish_reason": choice.finish_reason if choice else None,
                "usage": dict(completion.usage) if hasattr(completion, "usage") and completion.usage else None,
                "message_content": choice.message.content if choice else None,
                "tool_calls_count": len(choice.message.tool_calls or []) if choice else 0,
            })

            # Detect null/empty completion (upstream rate limit without a 429 status)
            if choice is None or (
                choice.finish_reason is None
                and not (choice.message.tool_calls or (choice.message.content or "").strip())
            ):
                wait = _BASE_DELAY * (2 ** _attempt)
                print(f"[DEBUG] Null completion at step {step} (attempt {_attempt+1}/{_MAX_RETRIES}) — waiting {wait}s", flush=True)
                import time; time.sleep(wait)
                continue  # retry

            # Native tool call (preferred)
            if choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"[DEBUG] Tool call: {tool_name}({list(args.keys())})", flush=True)
                vdump(f"TOOL CALL ARGS — {tool_name}", args)
                return {"tool": tool_name, "args": args}

            # Plain-text fallback (some providers ignore tool_choice="required")
            text = (choice.message.content or "").strip()
            print(f"[DEBUG] No tool_calls in response, trying text parse: {text[:100]}", flush=True)
            vprint(f"[VERBOSE] Full text response: {text}")
            return _parse_text_fallback(text, step, task_desc, app_base_url)

        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "429" in exc_str or "rate" in exc_str.lower() or "Rate" in exc_str
            if is_rate_limit and _attempt < _MAX_RETRIES - 1:
                wait = _BASE_DELAY * (2 ** _attempt)
                print(f"[DEBUG] Rate-limited at step {step} (attempt {_attempt+1}/{_MAX_RETRIES}) — waiting {wait}s then retrying", flush=True)
                import time; time.sleep(wait)
                continue  # retry
            # Non-rate-limit error or exhausted retries — don't call done(), keep episode alive
            print(f"[DEBUG] LLM call failed at step {step} (attempt {_attempt+1}): {exc}", flush=True)
            if step == 1:
                return {"tool": "browser_agent", "args": {"task": task_desc, "url": app_base_url}}
            return {"tool": "search_endpoints", "args": {"query": "available API endpoints"}}

    # Exhausted all retries — nudge forward without ending the episode
    print(f"[DEBUG] All {_MAX_RETRIES} retries exhausted at step {step} — nudging with search_endpoints", flush=True)
    if step == 1:
        return {"tool": "browser_agent", "args": {"task": task_desc, "url": app_base_url}}
    return {"tool": "search_endpoints", "args": {"query": "available API endpoints"}}


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
    # If the model explicitly says done, honour it — but only if text clearly indicates it.
    # A bare parse error should NEVER call done() because that would trigger the judge early.
    if re.search(r"\bdone\b", text.lower()) and len(text.strip()) < 80:
        return {"tool": "done", "args": {}}
    # Keep episode alive — nudge the model rather than punishing with a premature judge call.
    return {"tool": "search_endpoints", "args": {"query": "available REST API endpoints"}}


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
    task_params = dict(task_config.get("task_params") or {})

    # Pin the exact task so env.reset() uses the intended category/product instead
    # of sampling a random item from the template pool.
    os.environ["HARVGYM_TASK"] = str(template_id)
    os.environ["HARVGYM_TASK_SPEC_JSON"] = json.dumps(
        {
            "template_id": template_id,
            "description": task_description,
            "params": task_params,
            "base_url": app_base_url,
            "difficulty": task_config.get("difficulty", ""),
        }
    )

    env = HARvestGymEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # strict (0, 1) required by validator; 0.0 / 1.0 are rejected
    success = False
    last_result = None
    history: List[dict] = []
    session_state: dict = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        # Use the env-provided task description, which now matches the exact task spec
        # passed in above.
        task_desc = obs.task or task_description
        base_url = obs.app_base_url or app_base_url

        vprint(f"\n{'═'*60}")
        vprint(f"[VERBOSE] EPISODE START — {task_name}")
        vdump("INITIAL OBSERVATION (from env.reset)", obs.__dict__ if hasattr(obs, "__dict__") else str(obs))

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

                vprint(f"\n[VERBOSE] ── step {step} result ──")
                vdump(f"TOOL RESULT — {tool}", last_result)
                vprint(f"[VERBOSE] reward={reward:.3f}  done={done}")
                if done:
                    vdump("FINAL OBS (done=True)", obs.__dict__ if hasattr(obs, "__dict__") else str(obs))

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
                vprint(f"[VERBOSE] Step {step} EXCEPTION: {exc}")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        # Score: terminal reward from judge dominates.
        # Reward range by design: terminal success = +2 to +5, terminal fail = -1.5
        # Use a generous baseline so partial credit shows up.
        total_reward = sum(rewards)
        # Score: normalize to [0, 1] using per-template terminal-reward ceiling.
        # Template 1 (easy) max=2.0, Template 3 (medium) max=3.5, Template 6 (hard) max=5.0.
        # Shift by +1.5 so that the fail reward (-1.5) maps to 0 and max maps to 1.
        _TEMPLATE_REWARD_CEIL = {1: 2.0, 3: 3.5, 6: 5.0}
        _reward_ceil = _TEMPLATE_REWARD_CEIL.get(task_config.get("template_id"), 5.0)
        score = max(0.01, min(0.99, (total_reward + 1.5) / (_reward_ceil + 1.5)))
        success = total_reward >= 0.5   # any positive terminal reward = success

        vprint(f"\n[VERBOSE] ── episode end — {task_name} ──")
        vprint(f"[VERBOSE] total_reward={total_reward:.3f}  score={score:.3f}  success={success}")
        vprint(f"[VERBOSE] rewards per step: {[f'{r:.2f}' for r in rewards]}")

    except Exception as exc:
        error_str = str(exc)[:200]
        print(f"[DEBUG] Episode error: {error_str}", flush=True)
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_name": task_name,
        "difficulty": task_config.get("difficulty", "unknown"),
        "description": task_config.get("description", ""),
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
    for i, task_config in enumerate(TASKS, 1):
        difficulty = task_config.get("difficulty", "")
        desc = task_config.get("description", "")
        print(
            f"\n{'='*70}\n[TASK {i}/{len(TASKS)}] ({difficulty.upper()}) {desc}\n{'='*70}",
            flush=True,
        )
        result = await run_episode(task_config, client)
        results.append(result)
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"  → [{status}] score={result['score']:.2f}  steps={result['steps']}",
            flush=True,
        )

    # Summary grouped by difficulty tier
    print("\n" + "="*70, flush=True)
    print("[SUMMARY]", flush=True)
    for tier in ["easy", "medium", "hard"]:
        tier_results = [r for r in results if r.get("difficulty") == tier]
        if not tier_results:
            continue
        avg = sum(r["score"] for r in tier_results) / len(tier_results)
        passes = sum(1 for r in tier_results if r["success"])
        print(f"\n  {tier.upper()} ({passes}/{len(tier_results)} passed, avg score={avg:.2f}):", flush=True)
        for r in tier_results:
            status = "PASS" if r["success"] else "FAIL"
            print(f"    [{status}] {r['task_name']} — score={r['score']:.2f} steps={r['steps']}", flush=True)

    overall_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  OVERALL score={overall_score:.2f}  ({sum(1 for r in results if r['success'])}/{len(results)} passed)",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
