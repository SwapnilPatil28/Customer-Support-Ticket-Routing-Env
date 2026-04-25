import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from datasets import Dataset

from client import IncidentCommandEnvClient
from inference import HeuristicCoordinator, random_action
from models import IncidentAction


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_ROLLOUT_STEPS = int(os.getenv("MAX_ROLLOUT_STEPS", "120"))


@dataclass
class EpisodeStats:
    policy_name: str
    task_name: str
    total_reward: float
    steps: int
    success: bool


def obs_to_prompt(obs) -> str:
    return (
        "You are controlling a multi-agent incident command center.\n"
        f"Incident ID: {obs.incident_id}\n"
        f"Title: {obs.incident_title}\n"
        f"Description: {obs.incident_description}\n"
        f"Visible signals: {', '.join(obs.visible_signals)}\n"
        f"Budget remaining: {obs.budget_remaining}\n"
        f"SLA minutes remaining: {obs.sla_minutes_remaining}\n"
        f"Terminal output: {obs.terminal_output}\n"
        "Return a JSON object with keys: actor, action_type, target, root_cause, resolution_summary."
    )


def action_to_json(action: IncidentAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True)


def rollout(policy_name: str, task_name: str, collect_dataset: bool = False):
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    coordinator = HeuristicCoordinator()
    records: List[Dict[str, str]] = []
    rewards: List[float] = []
    steps = 0

    try:
        result = env.reset(task_name=task_name)
        while not result.done and steps < MAX_ROLLOUT_STEPS:
            steps += 1
            if policy_name == "heuristic":
                action = coordinator.select_action(result.observation)
            else:
                action = random_action(result.observation)

            if collect_dataset:
                records.append(
                    {
                        "prompt": obs_to_prompt(result.observation),
                        # TRL 0.20+ expects `completion` (not `response`) for prompt/completion SFT.
                        "completion": action_to_json(action),
                    }
                )

            result = env.step(action)
            rewards.append(float(result.reward or 0.0))
    finally:
        try:
            env.close()
        except Exception:
            pass

    total_reward = sum(rewards)
    success = total_reward > 0.0
    return EpisodeStats(policy_name, task_name, total_reward, steps, success), records, rewards


def build_training_dataset(episodes_per_task: int = 4) -> Dataset:
    all_rows: List[Dict[str, str]] = []
    for task in ["easy", "medium", "hard"]:
        for _ in range(episodes_per_task):
            _, rows, _ = rollout(policy_name="heuristic", task_name=task, collect_dataset=True)
            all_rows.extend(rows)
    return Dataset.from_list(all_rows)


def run_trl_sft(dataset: Dataset) -> None:
    """
    Minimal TRL script.
    This intentionally stays lightweight for CPU-friendly reproducibility.
    For actual hackathon runs, execute in Colab with a GPU and adjust params.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # TRL >= 0.20 uses `max_length`; older versions used `max_seq_length`.
    config = SFTConfig(
        output_dir="outputs/sft_run",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        max_length=768,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    # Use prompt + completion columns; pass tokenizer as processing_class (TRL 0.20+).
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()


def evaluate_policies() -> Dict[str, List[float]]:
    random_scores: List[float] = []
    heuristic_scores: List[float] = []

    for task in ["easy", "medium", "hard"]:
        random.seed(7)
        random_stats, _, _ = rollout("random", task)
        heuristic_stats, _, _ = rollout("heuristic", task)
        random_scores.append(random_stats.total_reward)
        heuristic_scores.append(heuristic_stats.total_reward)

    return {"random": random_scores, "heuristic": heuristic_scores}


def plot_rewards(score_map: Dict[str, List[float]]) -> None:
    labels = ["easy", "medium", "hard"]
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, score_map["random"], marker="o", label="Random baseline")
    plt.plot(x, score_map["heuristic"], marker="o", label="Heuristic coordinator")
    plt.xticks(x, labels)
    plt.xlabel("Task difficulty")
    plt.ylabel("Episode total reward")
    plt.title("Incident Command Center: baseline comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "reward_curve.png", dpi=160)
    plt.close()


def main() -> None:
    dataset = build_training_dataset(episodes_per_task=3)
    dataset.save_to_disk("artifacts/trl_dataset")

    run_trl_sft(dataset)
    scores = evaluate_policies()
    plot_rewards(scores)

    summary = {
        "base_model": BASE_MODEL,
        "dataset_rows": len(dataset),
        "random_rewards": scores["random"],
        "heuristic_rewards": scores["heuristic"],
    }
    with open(ARTIFACT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training and evaluation complete.")
    print(f"Saved artifacts in: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
