from __future__ import annotations
import os
import json
from typing import List, Optional

from openai import OpenAI
from env.environment import SpectreEnv
from agent.baseline_agent import BaselineAgent

# Environment variables (MUST use the hackathon-injected names — NO defaults)
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "spectre"
TASKS = ["easy", "medium", "hard", "expert"]
SEED = int(os.environ.get("SPECTRE_SEED", "42"))

# Initialize OpenAI client ONLY with hackathon-provided proxy values
client = None
if API_BASE_URL and API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] OpenAI client initialized with base_url={API_BASE_URL}", flush=True)
else:
    print(f"[DEBUG] WARNING: API_BASE_URL or API_KEY not set. LLM calls will be skipped.", flush=True)

# Baseline agent as fallback
baseline = BaselineAgent()


def safe_score(v: float) -> float:
    """Ensure score is STRICTLY between 0 and 1 (not 0.0, not 1.0)."""
    try:
        v = float(v)
    except:
        return 0.5
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return max(0.01, min(0.99, v))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    safe_reward = safe_score(reward)
    print(f"[STEP] step={step} action={action} reward={safe_reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = "true" if success else "false"
    safe_rewards = [safe_score(r) for r in rewards]
    safe_final_score = safe_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(f"[END] success={success_val} steps={steps} score={safe_final_score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(obs: dict) -> str:
    """Build LLM prompt from the current observation."""
    return f"""You are an AI agent solving a data pipeline task in the SPECTRE environment.

TASK: {obs['task']} - {obs['task_description']}

CURRENT STATE:
- Progress: {obs['progress']}/{obs['target_length']} steps completed
- Remaining: {obs['remaining_steps']} operations needed
- Next required operation: {obs['next_required_op']}
- Steps taken so far: {obs['step_count']}/{obs['max_steps']}
- Available primitives: {obs['available_primitives']}
- Custom tools defined: {obs['custom_tools_defined']}
- Tool registry: {json.dumps(obs['tool_registry'])}
- Compression ratio: {obs['compression_ratio']}

RULES:
1. Action types: "primitive" (execute one op), "create_tool" (define a reusable macro), "use_tool" (run a defined macro)
2. For "primitive": provide "name" from available_primitives matching next_required_op
3. For "create_tool": provide "name" and "sequence" (list of primitives or existing tools, min 2 items)
4. For "use_tool": provide "name" of an already-defined tool
5. GOAL: Complete the pipeline efficiently. Creating and reusing tools gives bonus rewards.

STRATEGY HINTS:
- For easy tasks: just use primitives directly
- For medium+: create an "etl_batch" tool = ["parse_data", "validate_data", "transform_data"], then reuse it
- For hard: create etl_batch, then create "triple_etl" = ["etl_batch", "etl_batch", "etl_batch"]
- For expert: create etl_batch, then "quad_etl" = ["etl_batch","etl_batch","etl_batch","etl_batch"]
- Use "aggregate_result" and "export_result" as final primitives when needed

Respond with ONLY a valid JSON object for the next action. Examples:
{{"type": "primitive", "name": "parse_data"}}
{{"type": "create_tool", "name": "etl_batch", "sequence": ["parse_data", "validate_data", "transform_data"]}}
{{"type": "use_tool", "name": "etl_batch"}}

Your JSON action:"""


def parse_llm_action(response_text: str) -> Optional[dict]:
    """Try to parse the LLM response as a JSON action."""
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except:
                continue
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except:
            pass
    
    return None


def get_llm_action(obs: dict) -> Optional[dict]:
    """Ask the LLM for the next action via the hackathon's API proxy."""
    if client is None:
        return None
    try:
        prompt = build_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert AI agent for data pipeline optimization. Respond with only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        text = response.choices[0].message.content
        print(f"[DEBUG] LLM response: {text}", flush=True)
        
        action = parse_llm_action(text)
        if action and "type" in action:
            return action
        
        print(f"[DEBUG] Could not parse LLM response as action", flush=True)
        return None
        
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return None


def run_task(task_name: str) -> None:
    """Run a single SPECTRE task using LLM-guided agent with baseline fallback."""
    env = SpectreEnv(task=task_name, seed=SEED, batch_file="orders_1.csv")
    baseline.reset()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset(seed=SEED)
        done = False
        
        while not done and steps_taken < env.max_steps:
            # Try LLM first
            action = get_llm_action(obs)
            
            # Fallback to baseline if LLM fails
            if action is None:
                print(f"[DEBUG] Using baseline fallback for step {steps_taken + 1}", flush=True)
                action = baseline.act(obs)
            
            # Execute step
            obs, reward, done, info = env.step(action)
            steps_taken += 1
            
            # Format action for logging
            action_type = action.get("type", "unknown")
            action_name = action.get("name", "")
            if action_type == "primitive":
                action_str = f"primitive({action_name})"
            elif action_type == "create_tool":
                action_str = f"create_tool({action_name})"
            elif action_type == "use_tool":
                action_str = f"use_tool({action_name})"
            else:
                action_str = str(action)
            
            error = info.get("error")
            rewards.append(reward)
            
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error
            )
            
            if done:
                break
        
        # Calculate final score
        total_reward = sum(rewards)
        max_possible = len(env.target_sequence) * 0.1
        score = total_reward / max(max_possible, 1.0) if max_possible > 0 else 0.0
        score = safe_score(score)
        
        success = obs["progress"] >= obs["target_length"]
        
    except Exception as e:
        print(f"[DEBUG] Task failed: {e}", flush=True)
        success = False
        score = 0.01
        if not rewards:
            rewards = [0.01]
    
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


def main() -> None:
    """Run all SPECTRE tasks."""
    for task in TASKS:
        try:
            run_task(task)
            print()
        except Exception as e:
            print(f"[ERROR] Task {task} crashed: {e}", flush=True)
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])
            print()


if __name__ == "__main__":
    main()
