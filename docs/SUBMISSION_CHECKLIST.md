# Submission Checklist — OpenEnv India 2026 Round 2

Status against every hard gate in the official judging rules, plus every polish item that moves the judging needle. **Last verified: all 21 tests passing, HF Space live, all artifacts committed.**

---

## Hard gates (from the official rules)

| # | Rule | Status | Evidence |
|---|---|---|---|
| 1 | **Use OpenEnv (latest release). Build on top of the framework; don't reinvent the wheel.** | ✅ | `requirements.txt` pins `openenv-core>=0.2.2`, `openenv.yaml` has `version: "3.0"`, `server/environment.py` extends `openenv.core.environment.Environment`, app built via `openenv.core.env_server.create_fastapi_app`. |
| 2 | **Working training script (Unsloth / HF TRL / any RL framework), ideally as a Colab notebook so judges can re-run it.** | ✅ | [`train_trl.py`](../train_trl.py) uses HF TRL `SFTTrainer`. **[One-click Colab notebook ↗](https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing)** runs the whole pipeline end-to-end on a T4 in ~1 h 15 min. |
| 3 | **Evidence that you actually trained: at minimum, loss and reward plots from a real run.** | ✅ | Four plots committed to [`artifacts/`](../artifacts): `training_curve.png` (loss + token accuracy), `reward_curve.png` (4-policy reward by tier), `reward_components.png` (per-component breakdown), plus the 0.5B ablation `reward_curve_qwen0p5b.png`. Full `training_log.json` + `summary_metrics.json` committed alongside. |
| 4 | **Short writeup or video: mini-blog on Hugging Face OR <2-min YouTube video, linked from README.** | ✅ | Mini-blog lives as [`docs/BLOG_POST.md`](./BLOG_POST.md) — shipped as part of the HF Space (rule 4 says "mini-blog on Hugging Face"; the Space is on HF and contains this file, so it renders at `huggingface.co/spaces/.../blob/main/docs/BLOG_POST.md`). All four training plots render inline via raw GitHub URLs. README and dashboard both link to it. (No separate video submission.) |
| 5 | **Push your environment to a Hugging Face Space so it's discoverable and runnable.** | ✅ | **Live at [`swapnilpatil28-multi-agent-incident-command-center.hf.space`](https://swapnilpatil28-multi-agent-incident-command-center.hf.space)** · Space page: [`huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center`](https://huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center). |
| 6 | **README motivates the problem, explains how the env works, and shows results.** | ✅ | [`README.md`](../README.md) — Part 1 ("Story in 2 minutes") opens with the problem in plain English, walks through the environment via role-permission tables, and shows all four plots + headline numbers. Part 2 is the full technical deep-dive (architecture, action/observation spaces, reward rubric, training pipeline, 0.5B ablation, ops/observability, testing, repo layout). |
| 7 | **README links to the HF Space + all additional materials (blog, slides, etc.).** | ✅ | "Live links" table inside Part 2 of the README lists every resource. Part 1 also has a "Try it in 30 seconds" CTA table. The dashboard header plus "Resources & documentation" grid surface the same links from the live Space itself. |
| 8 | **Do not include big video files in the HF submission — only public URLs.** | ✅ | No video files committed. All assets in [`artifacts/`](../artifacts) are PNG plots (≤ 162 KB each) + JSON. Repo weight is dominated by text and small images. |

---

## Judging-rubric alignment

### Environment Innovation (40%)

- [x] Multi-role, multi-agent — `triage_agent`, `investigator_agent`, `ops_manager_agent` with **non-overlapping permissions** (`server/domain/roles.py`).
- [x] Long-horizon — 3–5 sequential incidents per episode, 20–60 steps each, shared SLA + budget counters.
- [x] Professional / enterprise task simulation — realistic logs, metrics, KB articles, customer-tier revenue impact, SLA timers.
- [x] 13 unique incident templates across easy / medium / hard (`server/domain/incidents.py`).
- [x] Rich observation schema — customer tier, revenue impact, allowed actors per action, investigation targets grouped by tool, playbook hints, `reward_components`, `last_action_notes`.
- [x] Composable reward rubric with **14+ named components** and anti-gaming safeguards (`server/domain/reward.py`).
- [x] Tier-weighted business impact (`free ×0.6 · standard ×1.0 · premium ×1.4 · enterprise ×1.8`).
- [x] Role-based permissions + handoff scoring (`wrong_actor_penalty`, `handoff_correct`/`handoff_wrong`).

### Storytelling (30%)

- [x] README **Part 1 — The story in 2 minutes** written in plain English, readable by a non-technical judge in under 3 minutes.
- [x] Every plot has a one-line caption explaining what it shows.
- [x] Blog post [`docs/BLOG_POST.md`](./BLOG_POST.md) — eight labelled sections, four plots inline via raw GitHub URLs (render everywhere), 0.5B-vs-1.5B ablation narrative, explicit hackathon-theme mapping.
- [x] Live HF Space dashboard has a **"Story in 2 minutes"** hero panel at the top, a role-permission table, a three-card theme mapping, and a "Resources & documentation" grid with click-through links (README, blog, checklist, Colab, Space, etc.).
- [x] All documentation cross-links cleanly — README ↔ dashboard ↔ blog post ↔ checklist.

### Improvement in Rewards (20%)

- [x] 4-policy reward curve (`reward_curve.png`) across easy / medium / hard.
- [x] Training loss + token-accuracy curve (`training_curve.png`).
- [x] Reward-components stacked bar chart (`reward_components.png`) — shows *where* the improvement came from.
- [x] Ablation plot (`reward_curve_qwen0p5b.png`) for Qwen2.5-0.5B-Instruct backbone.
- [x] Per-task `improvement_sft_over_base` numbers in `summary_metrics.json`: **−1.80 / +3.13 / +10.17** (easy / medium / hard).
- [x] Final headline run: Qwen2.5-1.5B-Instruct, 8 episodes/task, 3 epochs, 680 rows — full `training_log.json` committed.

### Reward & Training Pipeline (10%)

- [x] Reward logic is coherent — rubric engine with module-level constants and unit tests (`tests/test_reward.py`).
- [x] Training pipeline genuinely connects to the running environment (no static dataset — rollouts collected from live `IncidentCommandCenterEnvironment`).
- [x] SFT checkpoint is saved to `artifacts/sft_model/` and reloaded for 4-policy evaluation — closes the loop.
- [x] 21 unit + integration tests passing (`tests/test_reward.py`, `tests/test_incidents.py`, `tests/test_environment.py`).

---

## Engineering table-stakes

- [x] Uses OpenEnv `Environment` base class properly.
- [x] Clean client/server separation — client only uses Pydantic models + HTTP (`client.py`).
- [x] Gym-style `reset / step / state` + OpenEnv `/close`.
- [x] Valid `openenv.yaml` manifest (version 3.0).
- [x] No reserved MCP tool names.
- [x] Structured JSON logging with per-episode seeded RNG (`server/logging_utils.py`).
- [x] Health / version / env-info / metrics endpoints (`/healthz`, `/version`, `/env-info`, `/metrics`).
- [x] Static `/artifacts` mount so the Space serves its own plots — no external hotlinking.
- [x] Dockerfile with `HEALTHCHECK` (`Dockerfile`, `server/Dockerfile`).
- [x] `pytest` passes cleanly: 21 / 21.
- [x] `.dockerignore` keeps image slim (excludes `sft_model/` checkpoint, keeps evidence plots).
- [x] `pre_validate.sh` + `validate-submission.sh` for one-command pre-submission smoke tests.
- [x] LICENSE (MIT) in repo root.

---

## Final submission steps

| # | Step | Status |
|---|---|---|
| 1 | Final training run (Qwen2.5-1.5B, 8 eps/task, 3 epochs) → all artifacts committed | ✅ |
| 2 | Commit artifacts (`reward_curve.png`, `training_curve.png`, `reward_components.png`, `reward_curve_qwen0p5b.png`, `training_log.json`, `summary_metrics.json`, `summary_metrics_qwen0p5b.json`) | ✅ |
| 3 | Update README with real numbers + real Space / Colab / GitHub / blog links | ✅ |
| 4 | Deploy HF Space from the same commit | ✅ |
| 5 | Dashboard upgraded: hero story panel, 4 stacked plots, resources grid with README / blog / checklist links | ✅ |
| 6 | Blog post updated (`docs/BLOG_POST.md`) with fixed image paths (raw GitHub URLs) and 0.5B ablation section | ✅ |
| 7 | All 21 tests passing on latest commit | ✅ |
| 8 | Run `openenv validate` remotely against the Space — `./validate-submission.sh <space-url>` | ⬜ (run it once before the deadline) |
| 9 | **Submit the Space URL in the hackathon form:** `https://swapnilpatil28-multi-agent-incident-command-center.hf.space` | ⬜ |
| 10 | Do not push commits after the submission deadline — post-deadline commits won't be considered | ⬜ |

---

## Pre-submission smoke test (copy-paste)

```bash
# 1. HF Space is serving
curl -fsS https://swapnilpatil28-multi-agent-incident-command-center.hf.space/healthz

# 2. Env-info endpoint advertises metadata
curl -s https://swapnilpatil28-multi-agent-incident-command-center.hf.space/env-info | head -20

# 3. OpenEnv validator passes remotely
./validate-submission.sh https://swapnilpatil28-multi-agent-incident-command-center.hf.space

# 4. A remote episode works
ENV_URL=https://swapnilpatil28-multi-agent-incident-command-center.hf.space python inference.py | head -40
```

## Where the judges will find each artefact

| Artefact | Primary URL |
|---|---|
| Live environment (OpenEnv-compatible) | [`swapnilpatil28-multi-agent-incident-command-center.hf.space`](https://swapnilpatil28-multi-agent-incident-command-center.hf.space) |
| Hugging Face Space page | [Space page ↗](https://huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center) |
| GitHub repository | [GitHub ↗](https://github.com/SwapnilPatil28/Multi-Agent-Incident-Command-Center) |
| README (Part 1 story + Part 2 deep-dive) | [`README.md`](../README.md) |
| Mini blog post (MD file in the repo, renders on both HF Space and GitHub) | [`docs/BLOG_POST.md`](./BLOG_POST.md) |
| Reproducible training notebook | [Colab ↗](https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing) |
| Training evidence (all 4 plots + JSON metrics) | [`artifacts/`](../artifacts) folder |
