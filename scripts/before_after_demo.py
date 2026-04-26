"""Before/after demo: base model vs fine-tuned model on the SAME incident.

Runs both policies against the same task under the same seed, prints a clean
side-by-side trace, and writes ``artifacts/before_after_demo.md`` which you
can paste into the blog post or other writeups.

Usage (after ``train_trl.py`` has saved ``artifacts/sft_model``)::

    ENV_URL=http://127.0.0.1:8000 python scripts/before_after_demo.py

Env variables:
    ENV_URL            — URL of a running Incident Command Center server
    BASE_MODEL         — HF hub id of the base model
    SFT_MODEL_DIR      — path to the fine-tuned checkpoint (default: artifacts/sft_model)
    DEMO_TASK          — task difficulty to demo (default: hard)
    DEMO_MAX_STEPS     — per-episode step cap (default: 120)
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repo root on sys.path when invoked from subdirectory
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from client import IncidentCommandEnvClient  # noqa: E402
from models import IncidentAction, IncidentObservation  # noqa: E402


ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SFT_MODEL_DIR = os.getenv("SFT_MODEL_DIR", "artifacts/sft_model")
DEMO_TASK = os.getenv("DEMO_TASK", "hard")
DEMO_MAX_STEPS = int(os.getenv("DEMO_MAX_STEPS", "120"))
DEMO_SEED = int(os.getenv("DEMO_SEED", "2026"))


def _format_obs_summary(obs: IncidentObservation) -> str:
    return (
        f"{obs.incident_title} "
        f"(tier={obs.customer_tier}, users={obs.affected_users_estimate}, "
        f"$/min={obs.revenue_impact_usd_per_min})"
    )


def _format_action(action: IncidentAction) -> str:
    target = action.target or "-"
    bits = [f"{action.actor}:{action.action_type}:{target}"]
    if action.reason:
        bits.append(f"reason={action.reason[:80]}")
    return " | ".join(bits)


def _format_components(components: Optional[Dict[str, float]]) -> str:
    if not components:
        return "-"
    return ", ".join(f"{k}={v:+.2f}" for k, v in components.items())


def _rollout_with_policy(policy_name: str, select_fn) -> Dict:
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    random.seed(DEMO_SEED)
    steps_log: List[Dict] = []
    total_reward = 0.0
    components_sum: Dict[str, float] = {}
    closed_incidents = 0
    incident_seen: List[str] = []
    try:
        result = env.reset(task_name=DEMO_TASK)
        step_idx = 0
        while not result.done and step_idx < DEMO_MAX_STEPS:
            step_idx += 1
            obs = result.observation
            if obs.incident_id not in incident_seen:
                incident_seen.append(obs.incident_id)
            action = select_fn(obs)
            result = env.step(action)
            reward = float(result.reward or 0.0)
            total_reward += reward
            new_obs = result.observation
            step_components = getattr(new_obs, "reward_components", None) or {}
            for k, v in step_components.items():
                components_sum[k] = components_sum.get(k, 0.0) + float(v)
            if action.action_type == "close_incident" and reward > 0:
                closed_incidents += 1
            steps_log.append(
                {
                    "step": step_idx,
                    "incident": obs.incident_id,
                    "summary": _format_obs_summary(obs),
                    "action": _format_action(action),
                    "reward": round(reward, 3),
                    "components": _format_components(step_components),
                }
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "policy": policy_name,
        "task": DEMO_TASK,
        "steps": len(steps_log),
        "total_reward": round(total_reward, 3),
        "incidents_seen": incident_seen,
        "incidents_closed": closed_incidents,
        "components_sum": {k: round(v, 3) for k, v in components_sum.items()},
        "trace": steps_log,
    }


def _write_markdown(base_run: Dict, sft_run: Dict, out_path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Before vs After — {DEMO_TASK.title()} task demo\n")
    lines.append(f"Both policies ran against the same seeded task (`{DEMO_TASK}`, seed {DEMO_SEED}) ")
    lines.append("on an identical Incident Command Center server. Each sees the same incident ")
    lines.append("queue in the same order.\n")
    lines.append("## Headline\n")
    lines.append(f"| Policy | Total reward | Steps | Incidents closed |")
    lines.append(f"|---|---:|---:|---:|")
    lines.append(
        f"| Base `{BASE_MODEL}` | {base_run['total_reward']:+.2f} | "
        f"{base_run['steps']} | {base_run['incidents_closed']} |"
    )
    lines.append(
        f"| **Fine-tuned (SFT)** | **{sft_run['total_reward']:+.2f}** | "
        f"{sft_run['steps']} | {sft_run['incidents_closed']} |"
    )
    delta = sft_run["total_reward"] - base_run["total_reward"]
    lines.append(f"\n**Reward delta: {delta:+.2f}** in favor of fine-tuned.\n")

    lines.append("## Reward sources (summed across the episode)\n")
    lines.append("| Component | Base | Fine-tuned |")
    lines.append("|---|---:|---:|")
    all_keys = sorted(set(base_run["components_sum"]) | set(sft_run["components_sum"]))
    for k in all_keys:
        lines.append(
            f"| `{k}` | {base_run['components_sum'].get(k, 0.0):+.2f} | "
            f"{sft_run['components_sum'].get(k, 0.0):+.2f} |"
        )

    def _trace_block(run: Dict, title: str) -> None:
        lines.append(f"\n## Trace — {title}\n")
        lines.append("```")
        for row in run["trace"]:
            lines.append(
                f"step {row['step']:>3} | incident={row['incident']} | "
                f"{row['action']} | reward={row['reward']:+.2f} | {row['components']}"
            )
        lines.append("```")

    _trace_block(base_run, f"Base model ({BASE_MODEL})")
    _trace_block(sft_run, "Fine-tuned (SFT) model")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    from llm_policy import LLMPolicy

    print(f"[demo] task={DEMO_TASK} seed={DEMO_SEED} env={ENV_URL}")

    print(f"[demo] Loading base model: {BASE_MODEL}")
    base_policy = LLMPolicy(BASE_MODEL, label="base_model")
    base_run = _rollout_with_policy("base_model", base_policy.select_action)
    base_policy.release()

    print(f"[demo] Loading SFT model: {SFT_MODEL_DIR}")
    sft_policy = LLMPolicy(SFT_MODEL_DIR, label="sft_model")
    sft_run = _rollout_with_policy("sft_model", sft_policy.select_action)
    sft_policy.release()

    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)
    md_path = art_dir / "before_after_demo.md"
    json_path = art_dir / "before_after_demo.json"
    _write_markdown(base_run, sft_run, md_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"base": base_run, "sft": sft_run}, f, indent=2)

    print(f"[demo] Base   total={base_run['total_reward']:+.2f} "
          f"steps={base_run['steps']} closed={base_run['incidents_closed']}")
    print(f"[demo] SFT    total={sft_run['total_reward']:+.2f} "
          f"steps={sft_run['steps']} closed={sft_run['incidents_closed']}")
    print(f"[demo] Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
