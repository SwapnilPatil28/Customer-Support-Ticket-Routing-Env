"""Hugging Face TRL training + evaluation pipeline.

Pipeline:

1. **Rollout**: run the ``HeuristicCoordinator`` against the live Incident
   Command Center environment to collect ``(prompt, completion)`` pairs.
2. **SFT**: fine-tune a small instruction-tuned LLM on those pairs using
   TRL's ``SFTTrainer`` with a single ``text`` column (robust across TRL
   ≥ 0.20).
3. **Save**: persist the fine-tuned weights + tokenizer to
   ``artifacts/sft_model`` so the same script can later load them as an
   agent policy.
4. **Evaluate**: play the environment with four policies
   ``random / heuristic / base_model / sft_model`` under identical seeds
   and write a reward curve + metrics JSON into ``artifacts/``.

Designed to work on CPU for smoke checks and on Colab T4 / HF Spaces GPUs
for full runs. LLM evaluation auto-enables on CUDA and can be forced with
``EVAL_LLM_MODELS=true``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from datasets import Dataset

from client import IncidentCommandEnvClient
from inference import HeuristicCoordinator, random_action
from models import IncidentAction, IncidentObservation


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SFT_MODEL_DIR = ARTIFACT_DIR / "sft_model"

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_ROLLOUT_STEPS = int(os.getenv("MAX_ROLLOUT_STEPS", "120"))
MAX_LLM_EVAL_STEPS = int(os.getenv("MAX_LLM_EVAL_STEPS", "60"))
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "3"))
TRAIN_EPOCHS = float(os.getenv("TRAIN_EPOCHS", "1"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
TRAIN_GRAD_ACCUM = int(os.getenv("TRAIN_GRAD_ACCUM", "2"))
TRAIN_MAX_LENGTH = int(os.getenv("TRAIN_MAX_LENGTH", "768"))
_EVAL_LLM_ENV = os.getenv("EVAL_LLM_MODELS", "auto").strip().lower()


@dataclass
class EpisodeStats:
    policy_name: str
    task_name: str
    total_reward: float
    steps: int
    success: bool


# ---------------------------------------------------------------------------
# Prompt / completion formatting
# ---------------------------------------------------------------------------


def obs_to_prompt(obs: IncidentObservation) -> str:
    targets = obs.investigation_targets or {}
    return (
        "You are operating a multi-agent incident command center. "
        "Pick the next action for the appropriate specialist role.\n\n"
        f"Incident ID: {obs.incident_id}\n"
        f"Title: {obs.incident_title}\n"
        f"Description: {obs.incident_description}\n"
        f"Customer tier: {obs.customer_tier} | "
        f"Affected users: {obs.affected_users_estimate} | "
        f"Revenue impact (USD/min): {obs.revenue_impact_usd_per_min}\n"
        f"Postmortem required: {obs.postmortem_required}\n"
        f"Visible signals: {', '.join(obs.visible_signals or [])}\n"
        f"Available log targets: {', '.join(targets.get('logs', []) or [])}\n"
        f"Available metric targets: {', '.join(targets.get('metrics', []) or [])}\n"
        f"Available KB articles: {', '.join(targets.get('kb', []) or [])}\n"
        f"Budget remaining: {obs.budget_remaining} actions | "
        f"SLA remaining: {obs.sla_minutes_remaining} min | "
        f"Clues found: {obs.clues_found} | "
        f"Mitigation applied: {obs.mitigation_applied}\n"
        f"Last terminal output: {obs.terminal_output}\n\n"
        "Respond with a JSON object containing exactly these keys: "
        "actor, action_type, target, root_cause, resolution_summary, "
        "postmortem_note, confidence, reason."
    )


def action_to_json(action: IncidentAction) -> str:
    payload = action.model_dump(exclude_none=True)
    return json.dumps(payload, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Rollout / dataset construction
# ---------------------------------------------------------------------------


def rollout(
    policy_name: str,
    task_name: str,
    collect_dataset: bool = False,
    policy_callable: Optional[Callable[[IncidentObservation], IncidentAction]] = None,
    max_steps: Optional[int] = None,
):
    """Play one episode and return (stats, rows, rewards).

    If ``policy_callable`` is provided it takes precedence over
    ``policy_name`` — this is how the LLM policies plug in.
    """
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    coordinator = HeuristicCoordinator()
    records: List[Dict[str, str]] = []
    rewards: List[float] = []
    steps = 0
    step_cap = max_steps if max_steps is not None else MAX_ROLLOUT_STEPS

    try:
        result = env.reset(task_name=task_name)
        while not result.done and steps < step_cap:
            steps += 1
            if policy_callable is not None:
                action = policy_callable(result.observation)
            elif policy_name == "heuristic":
                action = coordinator.select_action(result.observation)
            else:
                action = random_action(result.observation)

            if collect_dataset:
                records.append(
                    {
                        "prompt": obs_to_prompt(result.observation),
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
    return (
        EpisodeStats(policy_name, task_name, total_reward, steps, success),
        records,
        rewards,
    )


def build_training_dataset(episodes_per_task: int = EPISODES_PER_TASK) -> Dataset:
    rows: List[Dict[str, str]] = []
    for task in ["easy", "medium", "hard"]:
        for _ in range(episodes_per_task):
            _, new_rows, _ = rollout(
                policy_name="heuristic", task_name=task, collect_dataset=True
            )
            rows.extend(new_rows)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# TRL SFT
# ---------------------------------------------------------------------------


def _dataset_to_sft_text_column(dataset: Dataset, tokenizer) -> Dataset:
    """Collapse (prompt, completion) pairs into a single `text` field."""
    from transformers import PreTrainedTokenizerBase

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        return dataset

    cols = set(dataset.column_names)
    if "completion" not in cols and "response" in cols:
        dataset = dataset.rename_column("response", "completion")
    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        raise ValueError(
            f"Expected columns 'prompt' and 'completion' (or 'response'). "
            f"Got: {dataset.column_names}"
        )

    has_template = bool(getattr(tokenizer, "chat_template", None))

    def to_text_batched(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: List[str] = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            if has_template:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                out.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            else:
                out.append(f"User: {prompt}\n\nAssistant: {completion}")
        return {"text": out}

    to_drop = [c for c in dataset.column_names if c != "text"]
    return dataset.map(to_text_batched, batched=True, remove_columns=to_drop)


def run_trl_sft(dataset: Dataset) -> Path:
    """Fine-tune ``BASE_MODEL`` on the collected dataset and save the model.

    Returns the directory of the saved SFT checkpoint (``artifacts/sft_model``).
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
    train_ds = _dataset_to_sft_text_column(dataset, tokenizer)

    config = SFTConfig(
        output_dir="outputs/sft_run",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
        learning_rate=2e-5,
        num_train_epochs=TRAIN_EPOCHS,
        max_length=TRAIN_MAX_LENGTH,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    SFT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SFT_MODEL_DIR))
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))
    print(f"[train] Saved SFT checkpoint to {SFT_MODEL_DIR}")

    del trainer, model, tokenizer
    _free_gpu_memory()
    return SFT_MODEL_DIR


# ---------------------------------------------------------------------------
# Evaluation + reporting
# ---------------------------------------------------------------------------


def _free_gpu_memory() -> None:
    try:
        import gc
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _should_evaluate_llms() -> bool:
    if _EVAL_LLM_ENV in {"1", "true", "yes", "on"}:
        return True
    if _EVAL_LLM_ENV in {"0", "false", "no", "off"}:
        return False
    # "auto" / empty: enable only when a CUDA GPU is available so CPU runs
    # stay fast.
    return _cuda_available()


def _evaluate_single_policy(
    policy_name: str,
    select_fn: Callable[[IncidentObservation], IncidentAction],
    max_steps: Optional[int] = None,
) -> List[float]:
    scores: List[float] = []
    for task in ["easy", "medium", "hard"]:
        stats, _, _ = rollout(
            policy_name=policy_name,
            task_name=task,
            policy_callable=select_fn,
            max_steps=max_steps,
        )
        print(
            f"[eval] policy={policy_name} task={task} "
            f"reward={stats.total_reward:+.2f} steps={stats.steps}"
        )
        scores.append(round(stats.total_reward, 4))
    return scores


def evaluate_policies(
    seed: int = 7,
    evaluate_llms: Optional[bool] = None,
) -> Dict[str, List[float]]:
    """Run each policy once per task under the same seed.

    The random policy is seeded for reproducibility. The heuristic policy is
    deterministic already. LLM policies are evaluated with greedy decoding.
    """
    random.seed(seed)

    scores: Dict[str, List[float]] = {
        "random": [],
        "heuristic": [],
        "base_model": [],
        "sft_model": [],
    }

    for task in ["easy", "medium", "hard"]:
        random_stats, _, _ = rollout("random", task)
        heuristic_stats, _, _ = rollout("heuristic", task)
        scores["random"].append(round(random_stats.total_reward, 4))
        scores["heuristic"].append(round(heuristic_stats.total_reward, 4))

    should_eval_llms = _should_evaluate_llms() if evaluate_llms is None else evaluate_llms
    if not should_eval_llms:
        print("[eval] Skipping LLM evaluation (no GPU or EVAL_LLM_MODELS=false).")
        return scores

    try:
        from llm_policy import LLMPolicy
    except Exception as exc:  # pragma: no cover - import-time safety
        print(f"[eval] Could not import LLMPolicy ({exc}); skipping LLM eval.")
        return scores

    # Base model
    try:
        print(f"[eval] Loading BASE model: {BASE_MODEL}")
        base = LLMPolicy(BASE_MODEL, label="base_model")
        scores["base_model"] = _evaluate_single_policy(
            "base_model", base.select_action, max_steps=MAX_LLM_EVAL_STEPS
        )
        base.release()
        _free_gpu_memory()
    except Exception as exc:
        print(f"[eval] Base-model evaluation failed: {exc}")

    # SFT model
    if SFT_MODEL_DIR.exists():
        try:
            print(f"[eval] Loading SFT model: {SFT_MODEL_DIR}")
            sft = LLMPolicy(str(SFT_MODEL_DIR), label="sft_model")
            scores["sft_model"] = _evaluate_single_policy(
                "sft_model", sft.select_action, max_steps=MAX_LLM_EVAL_STEPS
            )
            sft.release()
            _free_gpu_memory()
        except Exception as exc:
            print(f"[eval] SFT-model evaluation failed: {exc}")
    else:
        print(f"[eval] No SFT checkpoint found at {SFT_MODEL_DIR}; skipping SFT eval.")

    return scores


def plot_rewards(score_map: Dict[str, List[float]]) -> None:
    labels = ["easy", "medium", "hard"]
    x = list(range(len(labels)))
    plt.figure(figsize=(9, 5))

    style = {
        "random": ("x", "tab:red", "Random baseline"),
        "heuristic": ("o", "tab:blue", "Heuristic coordinator"),
        "base_model": ("^", "tab:orange", "Base LLM (untrained)"),
        "sft_model": ("D", "tab:green", "Fine-tuned LLM (SFT)"),
    }

    for key, (marker, color, label) in style.items():
        values = score_map.get(key) or []
        if not values or len(values) != len(labels):
            continue
        plt.plot(x, values, marker=marker, color=color, label=label, linewidth=2)

    plt.xticks(x, labels)
    plt.xlabel("Task difficulty")
    plt.ylabel("Episode total reward")
    plt.title("Incident Command Center — policy comparison")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "reward_curve.png", dpi=160)
    plt.close()


def main() -> None:
    dataset = build_training_dataset(episodes_per_task=EPISODES_PER_TASK)
    dataset.save_to_disk(str(ARTIFACT_DIR / "trl_dataset"))

    run_trl_sft(dataset)
    scores = evaluate_policies()
    plot_rewards(scores)

    summary = {
        "base_model": BASE_MODEL,
        "dataset_rows": len(dataset),
        "episodes_per_task": EPISODES_PER_TASK,
        "random_rewards": scores.get("random", []),
        "heuristic_rewards": scores.get("heuristic", []),
        "base_model_rewards": scores.get("base_model", []),
        "sft_model_rewards": scores.get("sft_model", []),
        "improvement_sft_over_base": [
            round(s - b, 4)
            for s, b in zip(scores.get("sft_model", []), scores.get("base_model", []))
        ] if scores.get("sft_model") and scores.get("base_model") else [],
        "improvement_heuristic_over_random": [
            round(h - r, 4)
            for h, r in zip(scores.get("heuristic", []), scores.get("random", []))
        ],
    }
    with open(ARTIFACT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training and evaluation complete.")
    print(f"Saved artifacts in: {ARTIFACT_DIR.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
