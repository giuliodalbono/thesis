"""Every execution gets its own directory, including repeated same-seed runs."""
import json
from datetime import datetime, timezone
from uuid import uuid4


def new_run_directory(config, experiment):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    path = config.output_dir / config.environment_slug() / config.agent_name / experiment / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    # Checkpoint updates replace only this run's report, never previous runs.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
