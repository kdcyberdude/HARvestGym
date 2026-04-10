# HARvestGym — Build Notes & Deferred Items

This file captures caveats, deferred implementation decisions, and things to keep in mind during the building phase. It is not a specification — it is a living checklist.

---

## Critical Build-Time Checklist

### 1. `google/embeddinggemma-300m` — License Acceptance Required

**Status:** Deferred to build time  
**Action needed:** Accept the Google license for `google/embeddinggemma-300m` at https://huggingface.co/google/embeddinggemma-300m while logged in to HuggingFace. Then ensure `HF_TOKEN` is set in the environment before running any embedding code.

```bash
export HF_TOKEN=hf_...  # must have accepted the google/embeddinggemma-300m license
```

The model also requires `float32` or `bfloat16` — **not `float16`**. If you see activation errors, check the dtype:

```python
model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN)
# Default dtype is float32; explicitly set bfloat16 if on GPU:
# model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN, model_kwargs={"torch_dtype": "bfloat16"})
```

---

### 2. Judge Verification — Trajectory-Based, No External Token Needed

**Status:** Resolved in design  
**Detail:** The judge does **not** need a pre-set admin token or an outbound probe call to verify task completion. Verification is done by inspecting the episode trajectory already available in the environment.

**Approach:**
- The judge reads the `curl_exec` request/response history from the current episode (already stored in `episode.steps`).
- If the final state-changing call returned a 2xx response with the expected payload (e.g., `item_id` in add-to-cart, `order_id` in checkout, `post` in the forum response body), that response **is** the ground truth — the web server confirmed it.
- No re-probe is needed: the application already validated the request and returned success. The environment trusts that a 2xx from the live server is accurate.

**When a live probe is still used:** Template 3 (add-to-cart) and Template 7 (product creation) optionally re-fetch the resource (cart contents, product by SKU) to double-check state. These probes use the admin credentials the RL agent itself obtained during the episode (extracted from `session_state`), not a pre-configured environment token. If the agent did not authenticate (e.g., it tried to create a product without admin auth), the probe will 401 — which correctly scores the episode as failed.

**Implementation note for the judge:**
```python
# Prefer: check the response body from the agent's own curl calls
for step in episode.steps:
    if step.curl_parsed and step.curl_parsed.status_code == 200:
        body = step.curl_parsed.response_body
        # e.g. Template 3: look for item_id in add-to-cart response
        if isinstance(body, dict) and "item_id" in body:
            return 1.0

# Fallback live probe (optional, uses agent's own session token from episode):
admin_token = _extract_admin_token(episode)  # from agent's auth step
if admin_token:
    product = _judge_probe(f"GET /rest/V1/products/{sku}", base_url,
                           headers={"Authorization": f"Bearer {admin_token}"})
```

---

### 3. Forum (Postmill) — CSRF Token Position in HTML

**Status:** Handled in design; verify at build time  
**Detail:** The HTML truncation limit was raised to 3,000 characters specifically to capture hidden `<input type="hidden" name="_csrf_token">` fields. However, on some Postmill routes, the CSRF token appears after the main nav and the full form body. At build time, test the actual login page HTML to confirm the token appears within the first 3,000 characters.

```bash
curl -s 'http://ec2-...:9999/login' | head -c 3000 | grep _csrf_token
```

If the token is not captured, either:
- Increase `NONJSON_MAX_CHARS` further in `tools/curl_exec.py`
- Or rely on `search_episode_data("_csrf_token")` — the full HTML is indexed before truncation, so the token is always retrievable by keyword search regardless of position.

---

### 4. Wikipedia — HTML Wrapping

**Status:** Designed; implement in `curl_exec`  
**Detail:** Wikipedia (Kiwix) returns HTML. The environment wraps all non-JSON responses in a uniform JSON envelope `{status_code, headers, body}` before returning to the model. This wrapping is already part of `curl_exec`'s response structure (the `body` field is always a string for non-JSON content). No additional wrapping is needed — just ensure the system prompt tells the model to expect HTML strings in `body` for Wikipedia URLs.

---

### 5. Browser Agent — Deferred Live Implementation

**Status:** Deferred  
**Detail:** During training, `browser_agent` always loads from pre-recorded HAR files. The live browser agent (using Playwright + `browser-use/bu-30b-a3b-preview` served as a local service) is NOT needed for the initial training run.

At inference time, the live browser agent will be called as a separate service. The interface contract is:

```python
# The environment connects to the browser agent service via HTTP:
POST http://browser-agent-service/run
{"task": "...", "url": "..."}
→ {"app": "...", "endpoints": [...], "note": "..."}
```

Implementation details are in `BROWSER_AGENT.md`. Skip for now — HAR files cover the full training set.

---

### 6. OSM (Map Application) — Not In Initial Training Scope

**Status:** Intentionally excluded  
**Detail:** The OpenStreetMap application (port 3000) has ground truth catalogs and HAR recording tasks defined, but **no RL task templates target it in the initial training run**. The OSM artifacts are not needed for the first training loop.

Do not spend time on OSM tasks until the 7 current templates are training successfully.

---

### 7. `max_steps` Is 20, Not 12

**Status:** Updated in README and observation space  
**Reminder:** All code that initializes episodes must use `max_steps=20`. Search for any hardcoded `12` in the codebase before the first training run.

```bash
grep -r "max_steps.*12\|12.*max_steps" --include="*.py" .
```

---

### 8. GRPO Training Configuration

**Status:** Specified — follow the EcomRLVE-Gym pattern  
**Reference:** `winner_projects_last_ht/EcomRLVE-Gym/scripts/train_openenv.py` and `src/ecom_rlve/training/grpo.py`

**Stack:** Unsloth + TRL `GRPOTrainer`. The training script structure from EcomRLVE-Gym maps directly onto HARvestGym — replace the environment wrapper and reward functions, keep the training scaffolding.

**Policy model:** `Qwen/Qwen3-1.7B` with 4-bit quantization via Unsloth. LoRA rank 16, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

**Key configuration values (from EcomRLVE-Gym, adapted for HARvestGym):**

```python
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_length=8192,        # must fit 20-step Hard task episodes
    load_in_4bit=True,
    fast_inference=True,        # vLLM-backed fast generation for GRPO rollouts
    max_lora_rank=16,
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

training_args = GRPOConfig(
    num_generations=4,          # G=4 rollouts per prompt (EcomRLVE default; bump to 8 if VRAM allows)
    temperature=0.7,
    max_prompt_length=4096,     # auto-detect from dataset sample + 20% headroom
    max_completion_length=512,  # one tool call per step; curl commands are short
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=300,
    bf16=True,                  # use bfloat16 on Ampere+ GPUs
    output_dir="outputs/harvestgym_grpo",
)
```

**Reward functions passed to GRPOTrainer (three, like EcomRLVE-Gym):**
1. `format_reward` — does the output parse as a valid tool call? (`+1.0` / `-2.0`)
2. `tool_usage_reward` — is the tool name valid and arguments well-formed? (`+1.0` / `-0.5`)
3. `env_reward` — environment scalar reward from the judge, scaled ×5 to dominate (`-7.5` to `+25.0`)

**Curriculum:** Start all episodes on Template 1 (Easy). Introduce Medium templates when Easy success rate > 70%. Introduce Hard templates when Medium success rate > 60%.

**KL coefficient:** Start at `0.01`. If the model diverges from pretrained behavior rapidly (reward collapses after initial improvement), reduce to `0.005`.

---

### 9. System Prompt — Form vs JSON Guidance

**Status:** Designed in README; implement at build time  
**Detail:** The system prompt must include explicit instructions on when to use `Content-Type: application/x-www-form-urlencoded` vs `application/json`. Specifically:

```
For Postmill (Forum, port 9999): use form-encoded for login and post creation.
For Magento REST (Shopping/Admin, ports 7770/7780): use application/json.
For Wikipedia (port 8888): GET requests only, no Content-Type needed.
When in doubt: check the endpoint schema returned by search_endpoints() — it specifies the expected Content-Type.
```

---

### 10. HAR Is the Agent's Only API Knowledge Source — No Catalog Fallback

**Status:** Design decision, locked  
**Detail:** The `browser_agent` tool uses **only the HAR file** to build the agent's endpoint index and embeddings. The API catalogs (`catalogs/*.json`) are used exclusively by the judge for parameter-sourcing grading — they play no role in the training loop.

If a HAR yields very few endpoints, **the HAR recording needs to be improved**, not the code. The product does not patch sparse recordings by injecting catalog data into the agent's search corpus. This is intentional: the RL challenge is for the agent to discover and use APIs it has actually observed, not a curated ground-truth list.

**What goes where:**

| Data source | Who uses it | How |
|---|---|---|
| `hars/*.har` | Agent only | `browser_agent` → `search_endpoints` semantic search |
| `catalogs/*.json` | Judge only | Parameter-sourcing grading (`judge.py`) |

**Do not add catalog augmentation back** to `browser_agent.py` or `search_endpoints.py` under any circumstances. If the embed cache shows a large number of entries (e.g. 503 instead of 1), it means catalog entries leaked into the agent — clear the cache and fix the source.

---

## Non-Issues (Resolved in Design)

- ~~`store_finding` / `get_findings` tools~~ — **Removed**. Value threading happens through episode `history`.
- ~~`google/embeddinggemma-300m` doesn't exist~~ — **Confirmed real**. Uses `sentence-transformers` with `encode_query`/`encode_document`/`similarity`. Requires HF_TOKEN.
- ~~12 steps too few~~ — **Fixed to 20**.
- ~~Reward signal rewards busy episodes~~ — **Addressed** via curriculum learning + terminal reward dominance design. See README reward section.
- ~~Wikipedia task unwinnable~~ — **Resolved**: check for HTTP 200 + correct URL, not JSON content.
- ~~Forum CSRF handling~~ — **Resolved**: 3,000-char HTML truncation + `search_episode_data` fallback. No dedicated tool needed.
- ~~JUDGE_ADMIN_TOKEN expiry risk~~ — **Resolved**: judge reads trajectory response bodies directly; uses agent's own session token for optional probes only.
- ~~Concurrent episode isolation~~ — **Not needed**: multi-turn retry handles errors; no episode ID embedding required.
- ~~Parameter pool drift~~ — **Not a concern**: no training tasks involve deletion or reorganization; graders compare against expected values, not absolute DB state.
