"""LLM-backed policy for the Incident Command Center environment.

Wraps any Hugging Face causal-LM (a base model OR a fine-tuned checkpoint)
into a callable that takes an ``IncidentObservation`` and returns a typed
``IncidentAction``. This is what turns a raw language model into an agent
that can act inside the environment.

Usage::

    from llm_policy import LLMPolicy
    policy = LLMPolicy("Qwen/Qwen2.5-0.5B-Instruct")
    action = policy.select_action(observation)

If the model emits invalid JSON, the policy degrades gracefully to a safe
default action (inspect the first log target) so one bad generation never
crashes a whole rollout.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from models import IncidentAction, IncidentObservation

_LOG = logging.getLogger("icc.llm_policy")

# Regex for the first balanced-ish JSON object in the model output.
# (Greedy `.*` inside `{...}` keeps nested braces intact for our tiny JSON.)
_JSON_RE = re.compile(r"\{[\s\S]*\}")


class LLMPolicy:
    """Policy that calls a HF causal-LM and parses its JSON action."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        dtype: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime dep
            raise RuntimeError(
                "LLMPolicy requires `transformers` and `torch` installed. "
                "Run: pip install transformers torch"
            ) from exc

        self._torch = torch
        self.label = label or model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        _LOG.info(
            "Loading LLM policy %s on %s (dtype=%s)",
            model_name_or_path,
            resolved_device,
            torch_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(resolved_device)
        self.model.eval()
        self.device = resolved_device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, observation: IncidentObservation) -> IncidentAction:
        prompt_text = self._build_prompt_text(observation)
        response_text = self._generate(prompt_text)
        return self._parse_action(response_text, observation)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt_text(self, observation: IncidentObservation) -> str:
        # Keep this import here to avoid importing the trainer stack when the
        # module is used for inference only.
        from train_trl import obs_to_prompt

        user_prompt = obs_to_prompt(observation)
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {user_prompt}\n\nAssistant:"

    def _generate(self, prompt_text: str) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _parse_action(
        self,
        response_text: str,
        observation: IncidentObservation,
    ) -> IncidentAction:
        json_match = _JSON_RE.search(response_text)
        if json_match:
            raw = json_match.group(0)
            # Qwen / Llama sometimes add trailing commentary; strip past the
            # last closing brace to give JSON parser a clean slice.
            last_close = raw.rfind("}")
            if last_close != -1:
                raw = raw[: last_close + 1]
            try:
                data = json.loads(raw)
                return IncidentAction.model_validate(data)
            except Exception as exc:
                _LOG.debug(
                    "LLM JSON parse failed: %s :: raw=%s",
                    exc,
                    raw[:200],
                )

        return self._safe_fallback(observation)

    def _safe_fallback(self, observation: IncidentObservation) -> IncidentAction:
        logs = (observation.investigation_targets or {}).get("logs", []) or []
        target = logs[0] if logs else "payments-api"
        return IncidentAction(
            actor="triage_agent",
            action_type="inspect_logs",
            target=target,
            reason="LLM output invalid; using safe fallback action.",
        )

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free GPU memory so a second model can be loaded after this one."""
        try:
            import gc
            self.model = None  # type: ignore[assignment]
            self.tokenizer = None  # type: ignore[assignment]
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass
