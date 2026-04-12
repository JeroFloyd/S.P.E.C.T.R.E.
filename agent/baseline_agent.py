from __future__ import annotations

class BaselineAgent:
    def __init__(self):
        self._stage = 0

    def reset(self):
        self._stage = 0

    def act(self, obs: dict) -> dict:
        task = obs["task"]
        tools = set(obs["custom_tools_defined"])
        remaining = obs["remaining_steps"]
        next_op = obs.get("next_required_op")
        
        if task == "easy":
            return {"type": "primitive", "name": next_op}
        
        if task == "medium":
            if "etl_batch" not in tools:
                return {
                    "type": "create_tool",
                    "name": "etl_batch",
                    "sequence": ["parse_data", "validate_data", "transform_data"]
                }
            return {"type": "use_tool", "name": "etl_batch"}
        
        if task == "hard":
            if "etl_batch" not in tools:
                return {
                    "type": "create_tool",
                    "name": "etl_batch",
                    "sequence": ["parse_data", "validate_data", "transform_data"]
                }
            if "triple_etl" not in tools:
                return {
                    "type": "create_tool",
                    "name": "triple_etl",
                    "sequence": ["etl_batch", "etl_batch", "etl_batch"]
                }
            if remaining > 1:
                return {"type": "use_tool", "name": "triple_etl"}
            return {"type": "primitive", "name": "export_result"}
        
        if task == "expert":
            if "etl_batch" not in tools:
                return {
                    "type": "create_tool",
                    "name": "etl_batch",
                    "sequence": ["parse_data", "validate_data", "transform_data"]
                }
            if "quad_etl" not in tools:
                return {
                    "type": "create_tool",
                    "name": "quad_etl",
                    "sequence": ["etl_batch", "etl_batch", "etl_batch", "etl_batch"]
                }
            if remaining > 2:
                return {"type": "use_tool", "name": "quad_etl"}
            if remaining == 2:
                return {"type": "primitive", "name": "aggregate_result"}
            return {"type": "primitive", "name": "export_result"}
        
        return {"type": "primitive", "name": next_op}
