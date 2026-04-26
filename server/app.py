"""FastAPI entry-point for the Incident Command Center environment.

Besides the OpenEnv contract endpoints (`/reset`, `/step`, `/state`, `/close`)
registered by `create_fastapi_app`, this module exposes:

- `GET /` and `GET /web` — interactive HTML dashboard.
- `GET /healthz` — liveness / readiness probe for orchestrators.
- `GET /version` — build metadata.
- `GET /metadata` — static environment metadata (action space, reward model).
- `GET /metrics` — lightweight in-process counters (best-effort).

The dashboard is written inline so the environment ships as a single
directory and can be embedded in Hugging Face Spaces without extra assets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app

from models import IncidentAction, IncidentObservation
from server.config import EnvConfig
from server.domain import ALL_ACTIONS, ALL_ROLES, build_incident_library
from server.domain.reward import (
    CLOSURE_CORRECT_BASE,
    CLOSURE_WRONG_PENALTY,
    CLUE_REWARD,
    HANDOFF_CORRECT_REWARD,
    MITIGATION_CORRECT_REWARD,
    STEP_COST_INVESTIGATION,
    TIER_MULTIPLIER,
)
from server.environment import IncidentCommandCenterEnvironment
from server.logging_utils import configure_logging

_LOG = logging.getLogger("icc.app")
_CONFIG = EnvConfig.from_env()
configure_logging(level=_CONFIG.log_level, structured=_CONFIG.structured_logging)

# External URLs surfaced on the dashboard so judges can jump straight from
# the HF Space to the GitHub / Colab / docs / training artifacts.
GITHUB_URL = "https://github.com/SwapnilPatil28/Multi-Agent-Incident-Command-Center"
SPACE_PAGE_URL = "https://huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center"
SPACE_APP_URL = "https://swapnilpatil28-multi-agent-incident-command-center.hf.space"
COLAB_URL = "https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing"
# Dashboard doc links point at the Hugging Face Space copies of the docs (not
# GitHub) so a judge who opens the Space stays inside the HF ecosystem. The
# README on the Space page is rendered directly, so we point at the Space
# root for it; the other three open the HF file browser.
README_URL = f"{SPACE_PAGE_URL}/blob/main/README.md"
BLOG_POST_URL = f"{SPACE_PAGE_URL}/blob/main/docs/BLOG_POST.md"
SUBMISSION_CHECKLIST_URL = f"{SPACE_PAGE_URL}/blob/main/docs/SUBMISSION_CHECKLIST.md"

app = create_fastapi_app(
    IncidentCommandCenterEnvironment,
    IncidentAction,
    IncidentObservation,
)

# Serve the committed training-evidence artifacts (reward_curve.png,
# training_curve.png, reward_components.png, summary_metrics.json, ...)
# so the dashboard can embed them without depending on external hosts.
_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
if _ARTIFACTS_DIR.exists():
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(_ARTIFACTS_DIR)),
        name="artifacts",
    )


def _load_summary_metrics() -> Dict[str, Any]:
    """Best-effort load of the committed training results for the dashboard."""
    path = _ARTIFACTS_DIR / "summary_metrics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _resolve_environment() -> IncidentCommandCenterEnvironment | None:
    """Best-effort retrieval of the running environment instance.

    OpenEnv versions differ in where they stash the environment, so we try a
    few well-known attribute names before giving up.
    """
    for attr in ("environment", "env", "_environment"):
        env = getattr(app.state, attr, None)
        if env is not None:
            return env  # type: ignore[return-value]
    return None


def _metadata_payload() -> Dict[str, Any]:
    library = build_incident_library()
    return {
        "name": _CONFIG.name,
        "version": _CONFIG.version,
        "tasks": library.tasks(),
        "incidents_per_task": {
            task: len(library.templates_for(task)) for task in library.tasks()
        },
        "actions": list(ALL_ACTIONS),
        "roles": list(ALL_ROLES),
        "reward_model": {
            "step_cost_investigation": STEP_COST_INVESTIGATION,
            "clue_reward": CLUE_REWARD,
            "handoff_correct": HANDOFF_CORRECT_REWARD,
            "mitigation_correct": MITIGATION_CORRECT_REWARD,
            "closure_correct_base": CLOSURE_CORRECT_BASE,
            "closure_wrong": CLOSURE_WRONG_PENALTY,
            "tier_multiplier": TIER_MULTIPLIER,
        },
        "budgets": {
            "easy": _CONFIG.easy_budget,
            "medium": _CONFIG.medium_budget,
            "hard": _CONFIG.hard_budget,
        },
        "sla_minutes": {
            "easy": _CONFIG.easy_sla_minutes,
            "medium": _CONFIG.medium_sla_minutes,
            "hard": _CONFIG.hard_sla_minutes,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "name": _CONFIG.name,
            "version": _CONFIG.version,
        }
    )


@app.get("/version", response_class=JSONResponse)
async def version() -> JSONResponse:
    return JSONResponse(
        {
            "name": _CONFIG.name,
            "version": _CONFIG.version,
            "default_seed": _CONFIG.default_seed,
        }
    )


@app.get("/env-info", response_class=JSONResponse)
async def env_info() -> JSONResponse:
    """Rich metadata about the environment (rubric, budgets, taxonomy)."""
    return JSONResponse(_metadata_payload())


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    env = _resolve_environment()
    lines = [
        f'icc_info{{name="{_CONFIG.name}",version="{_CONFIG.version}"}} 1',
    ]
    if env is not None and env.state is not None:
        s = env.state
        lines += [
            f'icc_episode_step_total {s.step_count}',
            f'icc_cumulative_reward {s.cumulative_reward}',
            f'icc_incidents_resolved_total {s.incidents_resolved}',
            f'icc_incidents_failed_total {s.incidents_failed}',
            f'icc_budget_remaining {s.budget_remaining}',
            f'icc_sla_minutes_remaining {s.sla_minutes_remaining}',
            f'icc_current_incident_index {s.current_incident_index}',
        ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_dashboard_html())


def _dashboard_html() -> str:
    metadata_json = json.dumps(_metadata_payload(), indent=2)
    metrics = _load_summary_metrics()
    artifacts_available = _ARTIFACTS_DIR.exists() and (
        _ARTIFACTS_DIR / "reward_curve.png"
    ).exists()

    # --- Headline training numbers (1.5B SFT vs base, hard task) -------------
    base_rewards = metrics.get("base_model_rewards") or [0.0, 0.0, 0.0]
    sft_rewards = metrics.get("sft_model_rewards") or [0.0, 0.0, 0.0]
    improvement = metrics.get("improvement_sft_over_base") or [0.0, 0.0, 0.0]
    headline_delta = improvement[2] if len(improvement) >= 3 else 0.0

    def _fmt(val: Any) -> str:
        try:
            return f"{float(val):+.2f}"
        except (TypeError, ValueError):
            return "—"

    training_rows = "".join(
        f"<tr><td>{tier}</td><td>{_fmt(base_rewards[idx])}</td>"
        f"<td>{_fmt(sft_rewards[idx])}</td>"
        f"<td class='delta'>{_fmt(improvement[idx])}</td></tr>"
        for idx, tier in enumerate(("easy", "medium", "hard"))
        if idx < len(base_rewards)
    )

    # --- Training-evidence block (plots + caption) ---------------------------
    if artifacts_available:
        plots_html = """
    <h2>Training evidence</h2>
    <p class='sub'>
      Committed artifacts from the reference training run
      (Qwen2.5-1.5B-Instruct, 8 episodes/task, 3 epochs) plus the
      Qwen2.5-0.5B-Instruct ablation. Click any plot to open it full-size.
    </p>
    <div class='plots'>
      <figure>
        <a href='/artifacts/reward_curve.png' target='_blank' rel='noopener'>
          <img src='/artifacts/reward_curve.png' alt='Reward curve by policy (1.5B)' loading='lazy' />
        </a>
        <figcaption><strong>1.5B reward curve.</strong> Mean episodic reward per task tier
        across Random / Heuristic / Base-LLM / SFT-LLM. SFT matches the heuristic
        demonstrator across every tier and outperforms the untuned base by
        <strong>+{hard}</strong> on hard incidents.</figcaption>
      </figure>
      <figure>
        <a href='/artifacts/training_curve.png' target='_blank' rel='noopener'>
          <img src='/artifacts/training_curve.png' alt='SFT training loss and token accuracy (1.5B)' loading='lazy' />
        </a>
        <figcaption><strong>1.5B training curve.</strong> Supervised loss collapses from
        <code>~2.84 → ~0.02</code> and next-token accuracy climbs from
        <code>~0.49 → ~0.99</code> over three epochs on 680 rollout tokens.</figcaption>
      </figure>
      <figure>
        <a href='/artifacts/reward_components.png' target='_blank' rel='noopener'>
          <img src='/artifacts/reward_components.png' alt='Reward component decomposition (1.5B)' loading='lazy' />
        </a>
        <figcaption><strong>1.5B reward-component breakdown.</strong> SFT reproduces the
        heuristic's positive components (<code>clue_bonus</code>,
        <code>mitigation_correct</code>, <code>closure_correct</code>,
        <code>speed_bonus</code>) while the base model stalls on
        <code>step_cost</code> and SLA penalties.</figcaption>
      </figure>
      <figure>
        <a href='/artifacts/reward_curve_qwen0p5b.png' target='_blank' rel='noopener'>
          <img src='/artifacts/reward_curve_qwen0p5b.png' alt='Reward curve by policy (0.5B ablation)' loading='lazy' />
        </a>
        <figcaption><strong>0.5B ablation reward curve.</strong> Same pipeline, smaller
        backbone. SFT improves by only <strong>+0.43 / +0.14 / +0.00</strong> over base —
        the 0.5B model is too small to absorb the multi-step, role-gated policy.
        Scale is the story.</figcaption>
      </figure>
    </div>
    <p class='sub' style='margin-top:0.75rem'>
      Raw files:
      <a href='/artifacts/summary_metrics.json'>summary_metrics.json</a>
      ·
      <a href='/artifacts/training_log.json'>training_log.json</a>
      ·
      <a href='/artifacts/summary_metrics_qwen0p5b.json'>summary_metrics_qwen0p5b.json</a>
    </p>
""".format(hard=_fmt(headline_delta))
    else:
        plots_html = (
            "<h2>Training evidence</h2>"
            "<div class='card'><p class='sub'>Plots not bundled in this image. "
            "See the <a href='" + GITHUB_URL + "/tree/main/artifacts'>GitHub artifacts folder</a>.</p></div>"
        )

    # --- 0.5B ablation summary ----------------------------------------------
    ablation_html = """
    <h2>Ablation: model scale matters for imitation learning</h2>
    <div class='card'>
      <p class='sub'>
        Same pipeline, same data schema — only the base-model size differs. The 0.5B
        model cannot absorb the expert policy; 1.5B matches it exactly.
      </p>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Easy Δ</th><th>Medium Δ</th><th>Hard Δ</th>
              <th>Heuristic match?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Qwen2.5-0.5B-Instruct</td>
              <td>+0.43</td><td>+0.14</td><td class='delta'>+0.00</td>
              <td>No (stuck on step-cost)</td>
            </tr>
            <tr>
              <td><strong>Qwen2.5-1.5B-Instruct</strong></td>
              <td>-1.80</td><td>+3.13</td><td class='delta good'>+10.17</td>
              <td><strong>Yes (exact match)</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
"""

    # Theme mapping now lives in the top story block — keep this var empty
    # so the existing `{themes_html}` slot renders to nothing (no duplication).
    themes_html = ""

    # --- Reward-rubric details ----------------------------------------------
    reward_rubric_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{value}</td></tr>"
        for name, value in (
            ("step_cost", f"{STEP_COST_INVESTIGATION} per investigation step"),
            ("clue_reward", f"+{CLUE_REWARD} per new fact"),
            ("handoff_correct", f"+{HANDOFF_CORRECT_REWARD}"),
            ("mitigation_correct", f"+{MITIGATION_CORRECT_REWARD}"),
            ("closure_correct_base", f"+{CLOSURE_CORRECT_BASE} × tier multiplier"),
            ("closure_wrong", f"{CLOSURE_WRONG_PENALTY} × tier multiplier"),
        )
    )

    return f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>Incident Command Center | OpenEnv Dashboard</title>
  <style>
    :root {{
      --primary:#3b82f6; --accent:#22d3ee; --bg:#0f172a;
      --card:#111c31; --card-2:#152238; --text:#e2e8f0; --muted:#94a3b8;
      --good:#22c55e; --bad:#ef4444; --warn:#f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, 'Segoe UI', sans-serif;
      background: radial-gradient(1000px 600px at 10% -10%, #1e293b, var(--bg));
      color: var(--text); padding: 2rem; margin: 0; min-height: 100vh;
    }}
    header {{ display:flex; align-items:center; justify-content:space-between; max-width:1200px; margin:0 auto 1.5rem; flex-wrap:wrap; gap:1rem; }}
    .brand {{ display:flex; align-items:center; gap:0.75rem; }}
    .logo {{ width:44px; height:44px; border-radius:10px; background:linear-gradient(135deg,var(--primary),var(--accent)); }}
    h1 {{ font-size:1.6rem; margin:0; }}
    h2 {{ font-size:1.25rem; margin:1.8rem 0 0.6rem; color:#cbd5e1; }}
    .sub {{ color: var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap:1rem; max-width:1200px; margin:0 auto; }}
    .grid-3 {{ grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); }}
    .card {{ background: var(--card); border: 1px solid #1f2a44; padding: 1.25rem; border-radius: 14px; }}
    .card h3 {{ margin:0 0 0.5rem; font-size:1rem; color:#f1f5f9; }}
    .pill {{ display:inline-block; padding:2px 8px; margin:2px; border-radius:999px; background:#1e293b; border:1px solid #334155; color:#cbd5e1; font-size:0.78rem; }}
    .pill.cta {{ background:linear-gradient(135deg,var(--primary),var(--accent)); color:#0b1225; border-color:transparent; font-weight:600; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    code {{ background:#0b1225; border:1px solid #1f2a44; padding:2px 6px; border-radius:6px; color:#67e8f9; font-family:'JetBrains Mono', monospace; }}
    pre {{ background:#0b1225; border:1px solid #1f2a44; padding: 1rem; border-radius: 10px; color:#cbd5e1; overflow-x:auto; font-size:0.85rem; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .kpi {{ display:flex; flex-direction:column; gap:0.25rem; }}
    .kpi .num {{ font-size:1.6rem; font-weight:700; color:#f8fafc; }}
    .kpi .lbl {{ color: var(--muted); font-size:0.8rem; }}
    .kpi .num.good {{ color: var(--good); }}
    footer {{ max-width:1200px; margin:2rem auto 0; color:var(--muted); font-size:0.85rem; }}
    /* Training-evidence plots: one plot per row, centred, with a tighter
       max-width so the charts read as compact figures rather than banners.
       Click the image to open the full-resolution PNG in a new tab. */
    .plots {{ display:flex; flex-direction:column; gap:1.25rem; max-width:1200px; margin:0 auto; }}
    .plots figure {{ background: var(--card); border:1px solid #1f2a44; border-radius: 14px; padding: 1rem 1.25rem; margin:0; }}
    .plots figure a {{ display:block; }}
    .plots img {{
      width:100%; height:auto; display:block;
      max-width:720px; margin:0 auto;
      border-radius:10px; background:#0b1225;
      transition: transform 0.2s ease;
    }}
    .plots img:hover {{ transform: scale(1.01); }}
    .plots figcaption {{ color: var(--muted); font-size:0.9rem; margin-top:0.6rem; line-height:1.55; text-align:center; max-width:720px; margin-left:auto; margin-right:auto; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse: collapse; margin-top:0.5rem; font-size:0.9rem; }}
    th, td {{ padding:0.5rem 0.75rem; text-align:left; border-bottom:1px solid #1f2a44; }}
    th {{ color:#cbd5e1; font-weight:600; }}
    td.delta {{ font-weight:600; color:#f8fafc; }}
    td.delta.good {{ color: var(--good); }}
    .links {{ display:flex; flex-wrap:wrap; gap:0.5rem; }}

    /* "Story in 2 minutes" hero panel — plain-English summary for judges. */
    .hero-card {{
      background: linear-gradient(135deg, #0f2647 0%, #172a4a 60%, #1f2a44 100%);
      border: 1px solid #1f2a44; border-radius: 16px;
      padding: 1.75rem 1.75rem 1.5rem; margin: 0 auto 1.5rem;
      max-width: 1200px; box-shadow: 0 6px 30px rgba(34,211,238,0.08);
    }}
    .hero-card h2 {{ font-size:1.35rem; margin:0 0 0.4rem; color:#f1f5f9; }}
    .hero-card h3 {{ font-size:1rem; color:#e2e8f0; margin:0 0 0.3rem; }}
    .hero-card .lede {{
      font-size:1.02rem; line-height:1.6; color:#e2e8f0;
      background:#0b1225; border-left: 3px solid var(--accent);
      padding: 0.9rem 1.1rem; border-radius: 6px; margin: 0.3rem 0 0;
    }}
    .hero-card .lede strong {{ color:#f8fafc; }}
    .hero-card table {{ font-size:0.92rem; }}
    .hero-card .card {{ background: #0e1a30; }}

    /* "Resources & documentation" click-through cards. */
    .res-card {{
      display:block; color: var(--text); text-decoration:none;
      background: var(--card); border:1px solid #1f2a44; border-radius:12px;
      padding: 1rem 1.1rem;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .res-card:hover {{
      border-color: var(--accent); transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(34,211,238,0.12);
      text-decoration:none;
    }}
    .res-icon {{ font-size:1.6rem; line-height:1; margin-bottom:0.5rem; }}
    .res-title {{ font-weight:600; color:#f1f5f9; margin-bottom:0.2rem; }}
  </style>
</head>
<body>
  <header>
    <div class='brand'>
      <div class='logo'></div>
      <div>
        <h1>Incident Command Center</h1>
        <div class='sub'>OpenEnv · Multi-Agent · Long-Horizon · Professional-Task Simulation</div>
      </div>
    </div>
    <div class='links'>
      <a class='pill cta' href='{GITHUB_URL}' target='_blank' rel='noopener'>GitHub</a>
      <a class='pill cta' href='{COLAB_URL}' target='_blank' rel='noopener'>Open in Colab</a>
      <a class='pill cta' href='{README_URL}' target='_blank' rel='noopener'>README</a>
      <a class='pill cta' href='{BLOG_POST_URL}' target='_blank' rel='noopener'>Blog post</a>
      <a class='pill' href='{SPACE_PAGE_URL}' target='_blank' rel='noopener'>HF Space page</a>
      <span class='pill'>v{_CONFIG.version}</span>
      <span class='pill'>task: easy / medium / hard</span>
    </div>
  </header>

  <div class='container'>

    <!-- ============================================================ -->
    <!-- PART 1 — Plain-English story for non-technical judges        -->
    <!-- ============================================================ -->
    <div class='hero-card'>
      <h2 style='margin-top:0'>🚨 The story in 2 minutes</h2>
      <p class='lede'>
        When a real tech company has an outage, <strong>three people's phones
        buzz at once</strong> — a Triage engineer, an Investigator, and an Ops
        Manager. They have to cooperate under a ticking <strong>SLA clock</strong>,
        every action costs <strong>budget</strong>, and every wrong call costs
        <strong>real money</strong> (enterprise outages hurt ~3× more than free-tier).
        <br /><br />
        We built a simulator of that war room — and we fine-tuned an LLM to run it
        <strong>as well as the human expert</strong>.
      </p>

      <h3 style='margin-top:1.25rem'>What is the environment?</h3>
      <p class='sub' style='margin:0 0 0.75rem'>
        Three specialist agents with <strong>different permissions</strong> resolve
        a live queue of 13 realistic tech incidents across 3 difficulty tiers.
      </p>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr><th>Role</th><th>Can do</th><th>Cannot do</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>🔍 <strong>Triage</strong></td>
              <td>Pull logs · check metrics · consult KB</td>
              <td>Close a ticket</td>
            </tr>
            <tr>
              <td>🧪 <strong>Investigator</strong></td>
              <td>Apply a fix · roll back a deploy</td>
              <td>Escalate or file a post-mortem</td>
            </tr>
            <tr>
              <td>👷 <strong>Ops Manager</strong></td>
              <td>Escalate · file post-mortem · <strong>close the ticket</strong></td>
              <td>Apply a code fix</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3 style='margin-top:1.25rem'>What did the agent learn?</h3>
      <p class='sub' style='margin:0'>
        Not "pick the right label." It learned a whole workflow — dig up clues,
        hand off to the right specialist, apply the correct fix, respect the SLA,
        file the post-mortem, close the ticket. The rubric makes every piece of
        that workflow <em>visible</em> as a named reward component, so you can
        see <em>why</em> the agent earned (or lost) points at every step.
      </p>

      <h3 style='margin-top:1.25rem'>Why it matters for the 3 hackathon themes</h3>
      <div class='grid grid-3'>
        <div class='card'>
          <h3>🤝 Theme #1 — Multi-Agent</h3>
          <p class='sub'>
            Three distinct roles with <strong>non-overlapping permissions</strong>.
            Wrong-actor calls → <code>-0.08</code>. Correct handoff → <code>+0.15</code>.
            Cooperation is <em>trained</em>, not hard-coded.
          </p>
        </div>
        <div class='card'>
          <h3>⏱️ Theme #2 — Long-Horizon</h3>
          <p class='sub'>
            Each episode runs <strong>3–5 sequential incidents</strong> over 20–60
            steps with a single ticking SLA clock. Big rewards (+0.80 × tier) only
            fire after clues → fix → post-mortem. Sparse and delayed by design.
          </p>
        </div>
        <div class='card'>
          <h3>🏢 Theme #3 — Professional World-Model</h3>
          <p class='sub'>
            Real logs, metrics, KB articles, red-herring signals, customer tiers,
            SLA timers, revenue impact. Close an enterprise ticket wrong and it
            hurts ~3× what a free-tier one does.
          </p>
        </div>
      </div>

      <p class='sub' style='margin-top:1rem;font-style:italic'>
        ↓ Keep scrolling for the headline numbers, training plots, ablation, and
        the full rubric. Or jump straight to the
        <a href='{README_URL}' target='_blank' rel='noopener'>README</a> or the
        <a href='{BLOG_POST_URL}' target='_blank' rel='noopener'>blog post</a>.
      </p>
    </div>

    <!-- ============================================================ -->
    <!-- Resources & documentation — every link the judges need       -->
    <!-- ============================================================ -->
    <h2>Resources &amp; documentation</h2>
    <div class='grid grid-3'>
      <a class='res-card' href='{GITHUB_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>💻</div>
        <div class='res-title'>GitHub repository</div>
        <div class='sub'>Full source, tests, Dockerfile, CI-ready</div>
      </a>
      <a class='res-card' href='{SPACE_PAGE_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>🤗</div>
        <div class='res-title'>Hugging Face Space page</div>
        <div class='sub'>Repo view, build logs, discussions</div>
      </a>
      <a class='res-card' href='{SPACE_APP_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>🟢</div>
        <div class='res-title'>Live environment</div>
        <div class='sub'>You are here — OpenEnv endpoints live</div>
      </a>
      <a class='res-card' href='{COLAB_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>🎓</div>
        <div class='res-title'>Reproduce training (Colab T4)</div>
        <div class='sub'>One-click notebook, ~1 h wall clock</div>
      </a>
      <a class='res-card' href='{README_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>📖</div>
        <div class='res-title'>README (Part 1 + Part 2)</div>
        <div class='sub'>Story overview + full technical deep-dive</div>
      </a>
      <a class='res-card' href='{BLOG_POST_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>📝</div>
        <div class='res-title'>Mini blog post</div>
        <div class='sub'>The short writeup — MD file on the HF Space + GitHub</div>
      </a>
      <a class='res-card' href='{SUBMISSION_CHECKLIST_URL}' target='_blank' rel='noopener'>
        <div class='res-icon'>✅</div>
        <div class='res-title'>Submission checklist</div>
        <div class='sub'>Every judging rule → where to find the evidence</div>
      </a>
    </div>

    <h2>Headline results</h2>
    <div class='grid'>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>SFT reward lift on hard tasks</span>
          <span class='num good'>{_fmt(headline_delta)}</span>
          <span class='sub'>vs Qwen2.5-1.5B-Instruct base</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Heuristic-policy match</span>
          <span class='num'>Exact</span>
          <span class='sub'>SFT clones the demonstrator across every tier</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Scale ablation (hard Δ)</span>
          <span class='num'>0.5B → 1.5B</span>
          <span class='sub'>+0.00 → +10.17: capacity matters</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Training data</span>
          <span class='num'>680 rows</span>
          <span class='sub'>24 heuristic rollouts · 3 epochs</span>
        </div>
      </div>
    </div>

    <h2>Environment at a glance</h2>
    <div class='grid'>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Incidents in library</span>
          <span class='num' id='kpi-inc'>—</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Specialist roles</span>
          <span class='num'>3</span>
          <span class='sub'>triage · investigator · ops manager</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Reward components</span>
          <span class='num'>14+</span>
          <span class='sub'>rubric-based, transparent</span>
        </div>
      </div>
      <div class='card'>
        <div class='kpi'>
          <span class='lbl'>Seeded reproducibility</span>
          <span class='num'>Yes</span>
          <span class='sub'>default seed {_CONFIG.default_seed}</span>
        </div>
      </div>
    </div>

    <h2>1.5B SFT vs base (reference run)</h2>
    <div class='card'>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr>
              <th>Task tier</th><th>Base reward</th><th>SFT reward</th><th>Δ</th>
            </tr>
          </thead>
          <tbody>
            {training_rows}
          </tbody>
        </table>
      </div>
      <p class='sub' style='margin-top:0.75rem'>
        Numbers loaded live from
        <a href='/artifacts/summary_metrics.json'>summary_metrics.json</a>
        committed alongside this Space.
      </p>
    </div>

    {plots_html}

    {ablation_html}

    {themes_html}

    <h2>Endpoints</h2>
    <div class='card'>
      <p class='sub'>Standard OpenEnv contract plus operational endpoints.</p>
      <ul>
        <li><code>POST /reset</code> — start a new episode (task_name, seed).</li>
        <li><code>POST /step</code> — submit an IncidentAction.</li>
        <li><code>GET /state</code> — full environment state.</li>
        <li><code>GET /healthz</code> — liveness probe.</li>
        <li><code>GET /version</code> — build information.</li>
        <li><code>GET /env-info</code> — action space, reward model, budgets.</li>
        <li><code>GET /metrics</code> — Prometheus-style counters.</li>
        <li><code>GET /docs</code> — interactive OpenAPI documentation.</li>
        <li><code>GET /artifacts/…</code> — committed training plots &amp; metrics.</li>
      </ul>
    </div>

    <h2>Action space</h2>
    <div class='card'>
      {"".join(f"<span class='pill'>{a}</span>" for a in ALL_ACTIONS)}
      <p class='sub' style='margin-top:0.5rem'>
        Each action is gated by the acting role; wrong-actor calls are penalised.
      </p>
    </div>

    <h2>Reward model</h2>
    <div class='card'>
      <p>
        Composable rubric with anti-gaming safeguards. Every step returns a
        <code>reward_components</code> dictionary so training curves are
        interpretable. Closure rewards and SLA penalties are scaled by
        customer-tier multipliers:
      </p>
      <p>
        {"".join(f"<span class='pill'>{tier}: x{mult}</span>" for tier, mult in TIER_MULTIPLIER.items())}
      </p>
      <div class='table-wrap'>
        <table>
          <thead><tr><th>Component</th><th>Signal</th></tr></thead>
          <tbody>{reward_rubric_rows}</tbody>
        </table>
      </div>
      <p class='sub' style='margin-top:0.75rem'>
        Full rubric (invalid-action, repeated-lookup, rollback-effective,
        post-mortem-logged, etc.) is documented in the
        <a href='{GITHUB_URL}#reward-model' target='_blank' rel='noopener'>README</a>.
      </p>
    </div>

    <h2>Metadata</h2>
    <div class='card'>
      <pre id='metadata-json'>{metadata_json}</pre>
    </div>
  </div>

  <footer>
    <div>
      <strong>Incident Command Center v{_CONFIG.version}</strong> · Built on
      <a href='https://github.com/meta-pytorch/openenv' target='_blank' rel='noopener'>OpenEnv</a>
      for the OpenEnv India 2026 Round 2 hackathon.
    </div>
    <div style='margin-top:0.4rem'>
      <a href='{GITHUB_URL}' target='_blank' rel='noopener'>GitHub</a> ·
      <a href='{SPACE_PAGE_URL}' target='_blank' rel='noopener'>HF Space page</a> ·
      <a href='{COLAB_URL}' target='_blank' rel='noopener'>Colab</a> ·
      <a href='{README_URL}' target='_blank' rel='noopener'>README</a> ·
      <a href='{BLOG_POST_URL}' target='_blank' rel='noopener'>Blog post</a> ·
      <a href='{SUBMISSION_CHECKLIST_URL}' target='_blank' rel='noopener'>Submission checklist</a>
    </div>
  </footer>

  <script>
    try {{
      const data = {metadata_json};
      const total = Object.values(data.incidents_per_task || {{}}).reduce((a,b)=>a+b,0);
      document.getElementById('kpi-inc').textContent = total;
    }} catch (e) {{}}
  </script>
</body>
</html>
"""


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
