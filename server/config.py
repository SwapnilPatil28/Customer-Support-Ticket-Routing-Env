"""Runtime configuration for the Incident Command Center environment.

All tunables are read from environment variables so the server is 12-factor
compatible and can be reconfigured per deployment without rebuilding the
image. Every field has a sensible default so local development "just works".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_VERSION = "3.0.0"
ENV_NAME = "incident_command_center_env"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EnvConfig:
    name: str = ENV_NAME
    version: str = ENV_VERSION

    default_seed: int = 20260425
    easy_budget: int = 28
    medium_budget: int = 54
    hard_budget: int = 84
    easy_sla_minutes: int = 120
    medium_sla_minutes: int = 210
    hard_sla_minutes: int = 330

    sla_tick_minutes: int = 5
    max_reward_trace_len: int = 400
    structured_logging: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "EnvConfig":
        return cls(
            name=os.getenv("ENV_NAME", ENV_NAME),
            version=os.getenv("ENV_VERSION", ENV_VERSION),
            default_seed=_int_env("ENV_SEED", 20260425),
            easy_budget=_int_env("ENV_EASY_BUDGET", 28),
            medium_budget=_int_env("ENV_MEDIUM_BUDGET", 54),
            hard_budget=_int_env("ENV_HARD_BUDGET", 84),
            easy_sla_minutes=_int_env("ENV_EASY_SLA", 120),
            medium_sla_minutes=_int_env("ENV_MEDIUM_SLA", 210),
            hard_sla_minutes=_int_env("ENV_HARD_SLA", 330),
            sla_tick_minutes=_int_env("ENV_SLA_TICK", 5),
            max_reward_trace_len=_int_env("ENV_MAX_REWARD_TRACE_LEN", 400),
            structured_logging=_bool_env("ENV_STRUCTURED_LOGGING", True),
            log_level=os.getenv("ENV_LOG_LEVEL", "INFO"),
        )

    def budget_for(self, task_name: str) -> int:
        return {
            "easy": self.easy_budget,
            "medium": self.medium_budget,
            "hard": self.hard_budget,
        }.get(task_name, self.medium_budget)

    def sla_for(self, task_name: str) -> int:
        return {
            "easy": self.easy_sla_minutes,
            "medium": self.medium_sla_minutes,
            "hard": self.hard_sla_minutes,
        }.get(task_name, self.medium_sla_minutes)
