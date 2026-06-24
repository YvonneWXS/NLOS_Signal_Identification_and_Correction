# common/config_manager.py — YAML config loading with CLI override
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional


def load_config(config_path: Union[str, Path], cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config and merge CLI overrides (dot-separated keys)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if cli_overrides:
        for key, value in cli_overrides.items():
            _set_nested(config, key.split("."), value)
    return config


def _set_nested(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def save_condition_md(config: Dict, output_dir: Union[str, Path], extra: Optional[Dict] = None):
    """Save experiment condition record as condition.md."""
    import sys, platform
    from datetime import datetime
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment Condition Record", ""]
    lines.append(f"**Timestamp**: {datetime.now().isoformat()}")
    lines.append(f"**Host**: {platform.node()}")
    lines.append(f"**Python**: {sys.version.split()[0]}")
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        lines.append(f"**Git commit**: {repo.head.object.hexsha[:8]}")
        lines.append(f"**Git branch**: {repo.active_branch.name}")
    except Exception:
        lines.append("**Git**: unavailable")
    try:
        import torch
        lines.append(f"**PyTorch**: {torch.__version__}")
        lines.append(f"**CUDA available**: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"**GPU**: {torch.cuda.get_device_name(0)}")
    except Exception:
        lines.append("**PyTorch**: unavailable")
    lines.append("")
    lines.append("## Configuration")
    lines.append("```yaml")
    lines.append(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    lines.append("```")
    if extra:
        lines.append("")
        lines.append("## Additional Info")
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
    with open(output_dir / "condition.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
