from __future__ import annotations

import uuid
from pathlib import Path

from env.tasks import TASK_REGISTRY
from env.actions import validate_action, PRIMITIVES
from env.rewards import compute_reward
from env.pipeline import PipelineState, parse, validate, transform, aggregate, export


class SpectreEnv:
    def __init__(self, task: str = "medium", seed: int | None = None, batch_file: str = "orders_1.csv"):
        self.task_name = task
        self.seed = seed
        self.batch_file = batch_file  # CRITICAL: specific batch to process
        self.session_id = str(uuid.uuid4())
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        task = TASK_REGISTRY[self.task_name]
        self.target_sequence = task["sequence"]
        self.max_steps = task["max_steps"]
        self.task_description = task["description"]
        self.current_index = 0
        self.progress = 0
        self.prev_progress = 0
        self.step_count = 0
        self.custom_tools = {}
        self.tool_registry = {}
        self._step_log = []
        self._pipeline = PipelineState(
            task=self.task_name,
            data_dir=Path("data"),
            seed=seed or 42,
            batch_file=self.batch_file  # PASS selected batch
        )
        return self.state()

    def state(self):
        remaining = len(self.target_sequence) - self.current_index
        next_op = self.target_sequence[self.current_index] if self.current_index < len(self.target_sequence) else None
        compression = round(self.progress / self.step_count, 3) if self.step_count > 0 else 0.0
        return {
            "task": self.task_name,
            "task_description": self.task_description,
            "session_id": self.session_id,
            "progress": self.progress,
            "target_length": len(self.target_sequence),
            "remaining_steps": remaining,
            "next_required_op": next_op,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "available_primitives": PRIMITIVES,
            "available_tools": list(self.custom_tools.keys()),
            "custom_tools_defined": list(self.custom_tools.keys()),
            "tool_registry": self.tool_registry,
            "compression_ratio": compression,
            "pipeline_state": self._pipeline.summary(),
        }

    def step(self, action: dict):
        self.step_count += 1
        self.prev_progress = self.progress
        error = None

        err = validate_action(action, list(self.custom_tools.keys()))
        if err:
            return self.state(), -0.05, False, {"error": err}

        try:
            if action["type"] == "primitive":
                self._apply_primitive(action["name"])
            elif action["type"] == "create_tool":
                name = action["name"]
                sequence = action["sequence"]
                self.custom_tools[name] = sequence
                self.tool_registry[name] = {"sequence": sequence, "expanded_length": self._expand_length(sequence)}
            elif action["type"] == "use_tool":
                self._execute_tool(action["name"])
        except Exception as exc:
            error = str(exc)

        done = self.current_index >= len(self.target_sequence)
        reward, breakdown = compute_reward(
            step_count=self.step_count, max_steps=self.max_steps, done=done,
            progress=self.progress, target_length=len(self.target_sequence),
            prev_progress=self.prev_progress, pipeline=self._pipeline, custom_tools=self.custom_tools,
        )

        info = {"error": error, "reward_breakdown": breakdown, "action": action, "session_id": self.session_id}
        self._step_log.append({"action": action, "reward": reward, "info": info})
        return self.state(), reward, done, info

    def _apply_primitive(self, name: str):
        if self.current_index >= len(self.target_sequence):
            return
        expected = self.target_sequence[self.current_index]
        if name != expected:
            return
        self.current_index += 1
        self.progress += 1
        dispatch = {
            "parse_data": lambda: parse(self._pipeline),
            "validate_data": lambda: validate(self._pipeline),
            "transform_data": lambda: transform(self._pipeline),
            "aggregate_result": lambda: aggregate(self._pipeline),
            "export_result": lambda: export(self._pipeline),
        }
        fn = dispatch.get(name)
        if fn:
            fn()

    def _execute_tool(self, name: str):
        for step in self.custom_tools.get(name, []):
            self._apply_step_safe(step)

    def _apply_step_safe(self, step: str, depth: int = 0):
        if depth > 10:
            return
        if step in self.custom_tools:
            for sub in self.custom_tools[step]:
                self._apply_step_safe(sub, depth + 1)
        else:
            self._apply_primitive(step)

    def _expand_length(self, sequence: list[str]) -> int:
        total = 0
        for step in sequence:
            if step in self.custom_tools:
                total += self._expand_length(self.custom_tools[step])
            else:
                total += 1
        return total
