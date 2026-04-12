from __future__ import annotations

TASK_REGISTRY: dict[str, dict] = {
    "easy": {
        "sequence": ["parse_data", "validate_data", "transform_data"],
        "max_steps": 20,
        "description": "Single batch ETL. Optimal: 3 steps (just execute primitives).",
    },
    "medium": {
        "sequence": ["parse_data", "validate_data", "transform_data", 
                     "parse_data", "validate_data", "transform_data"],
        "max_steps": 30,
        "description": "TWO batch ETL. Optimal: 3 steps (create etl_batch tool + use it 2x).",
    },
    "hard": {
        "sequence": ["parse_data", "validate_data", "transform_data", 
                     "parse_data", "validate_data", "transform_data",
                     "parse_data", "validate_data", "transform_data",
                     "export_result"],
        "max_steps": 50,
        "description": "THREE batch ETL + export. Optimal: 5 steps (create tools + compress).",
    },
    "expert": {
        "sequence": ["parse_data", "validate_data", "transform_data",
                     "parse_data", "validate_data", "transform_data",
                     "parse_data", "validate_data", "transform_data",
                     "parse_data", "validate_data", "transform_data",
                     "aggregate_result", "export_result"],
        "max_steps": 60,
        "description": "FOUR batch ETL + aggregate + export. Optimal: 7 steps (hierarchical tools).",
    },
}
