from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair PEFT LoRA adapter config exported with zero scaling.")
    parser.add_argument("--adapter-path", required=True, help="Directory containing adapter_config.json.")
    parser.add_argument("--alpha", type=int, default=0, help="LoRA alpha to set. Defaults to rank r.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended changes without writing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    adapter_path = Path(args.adapter_path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = _positive_int(config.get("r"), 32)
    alpha = args.alpha if args.alpha > 0 else rank

    before = {
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "alpha_pattern": config.get("alpha_pattern"),
    }

    changed = False
    if _positive_int(config.get("lora_alpha"), 0) == 0:
        config["lora_alpha"] = alpha
        changed = True

    alpha_pattern = config.get("alpha_pattern")
    if isinstance(alpha_pattern, dict):
        repaired = {}
        for key, value in alpha_pattern.items():
            repaired[key] = alpha if _positive_int(value, 0) == 0 else value
        if repaired != alpha_pattern:
            config["alpha_pattern"] = repaired
            changed = True

    after = {
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "alpha_pattern": config.get("alpha_pattern"),
    }

    print(json.dumps({"path": str(config_path), "changed": changed, "before": before, "after": after}, indent=2))

    if changed and not args.dry_run:
        backup_path = config_path.with_suffix(".json.bak")
        if not backup_path.exists():
            backup_path.write_text(json.dumps(json.loads(config_path.read_text(encoding="utf-8")), indent=2) + "\n")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
