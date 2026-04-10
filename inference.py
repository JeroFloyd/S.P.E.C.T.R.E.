"""
S.P.E.C.T.R.E Inference Script
===================================
OpenEnv-compliant stdout format:
    [START] task=<task> env=spectre model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00>
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from env.environment      import SpectreEnv
from agent.baseline_agent import BaselineAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASKS        = ["easy", "medium", "hard"]
SEED         = int(os.getenv("SPECTRE_SEED", "42"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """\
You are an autonomous agent in S.P.E.C.T.R.E. Complete the data pipeline in as few steps as possible.
Respond with ONLY a valid JSON object. No explanation, no markdown.\
"""


def get_llm_action(obs: dict) -> dict | None:
    if client is None:
        return None
    try:
        prompt = (
            f"Task: {obs['task']} | Next op: {obs['next_required_op']} | "
            f"Remaining: {obs['remaining_steps']} | Tools: {obs['custom_tools_defined']}\n"
            f"Choose your next action as a single JSON object."
        )
        resp = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0,
            max_tokens  = 200,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


def _safe(v: float) -> float:
    """Strictly clamp to (0.01, 0.99) — never 0.0 or 1.0."""
    v = float(v)
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return min(0.99, max(0.01, v))


def _compute_score(rewards: list[float], success: bool, steps: int, max_steps: int) -> float:
    """Compute a single task score strictly within (0.01, 0.99)."""
    if not rewards:
        return 0.01
    mean_r = sum(rewards) / len(rewards)
    efficiency = max(0.01, 1.0 - (steps / max(max_steps, 1)))
    if success:
        score = 0.5 * mean_r + 0.5 * efficiency
    else:
        score = 0.2 * mean_r
    return _safe(score)


def log_start(task: str, model: str):
    print(f"[START] task={task} env=spectre model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={_safe(reward):.4f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float):
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_safe(score):.4f}",
        flush=True,
    )


def run_task(task_name: str):
    env      = SpectreEnv(task=task_name, seed=SEED)
    fallback = BaselineAgent()

    obs      = env.reset(seed=SEED)
    done     = False
    rewards: list[float] = []
    steps    = 0
    success  = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        while not done and steps < env.max_steps:
            steps += 1
            try:
                action = get_llm_action(obs) if client else None
                if action is None:
                    action = fallback.act(obs)

                obs, reward, done, info = env.step(action)
                reward = _safe(reward)
                rewards.append(reward)
                log_step(steps, action, reward, done, info.get("error"))

            except Exception as exc:
                log_step(steps, {"error": str(exc)}, 0.01, True, str(exc))
                done = True

        success = obs["progress"] >= obs["target_length"]

    except Exception:
        success = False

    finally:
        score = _compute_score(rewards, success, steps, env.max_steps)
        log_end(success=success, steps=steps, score=score)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        print()
