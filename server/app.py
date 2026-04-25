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
from typing import Any, Dict

import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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

app = create_fastapi_app(
    IncidentCommandCenterEnvironment,
    IncidentAction,
    IncidentObservation,
)


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
    header {{ display:flex; align-items:center; justify-content:space-between; max-width:1100px; margin:0 auto 1.5rem; }}
    .brand {{ display:flex; align-items:center; gap:0.75rem; }}
    .logo {{ width:44px; height:44px; border-radius:10px; background:linear-gradient(135deg,var(--primary),var(--accent)); }}
    h1 {{ font-size:1.6rem; margin:0; }}
    h2 {{ font-size:1.1rem; margin:1.4rem 0 0.6rem; color:#cbd5e1; }}
    .sub {{ color: var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:1rem; max-width:1100px; margin:0 auto; }}
    .card {{ background: var(--card); border: 1px solid #1f2a44; padding: 1.25rem; border-radius: 14px; }}
    .card h3 {{ margin:0 0 0.5rem; font-size:1rem; color:#f1f5f9; }}
    .pill {{ display:inline-block; padding:2px 8px; margin:2px; border-radius:999px; background:#1e293b; border:1px solid #334155; color:#cbd5e1; font-size:0.78rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    code {{ background:#0b1225; border:1px solid #1f2a44; padding:2px 6px; border-radius:6px; color:#67e8f9; font-family:'JetBrains Mono', monospace; }}
    pre {{ background:#0b1225; border:1px solid #1f2a44; padding: 1rem; border-radius: 10px; color:#cbd5e1; overflow-x:auto; font-size:0.85rem; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .kpi {{ display:flex; flex-direction:column; gap:0.25rem; }}
    .kpi .num {{ font-size:1.6rem; font-weight:700; color:#f8fafc; }}
    .kpi .lbl {{ color: var(--muted); font-size:0.8rem; }}
    footer {{ max-width:1100px; margin:2rem auto 0; color:var(--muted); font-size:0.85rem; }}
  </style>
</head>
<body>
  <header>
    <div class='brand'>
      <div class='logo'></div>
      <div>
        <h1>Incident Command Center</h1>
        <div class='sub'>OpenEnv · Multi-Agent · Long-Horizon · Enterprise Simulation</div>
      </div>
    </div>
    <div>
      <span class='pill'>v{_CONFIG.version}</span>
      <span class='pill'>task: easy / medium / hard</span>
    </div>
  </header>

  <div class='container'>
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
      </ul>
    </div>

    <h2>Action space</h2>
    <div class='card'>
      {"".join(f"<span class='pill'>{a}</span>" for a in ALL_ACTIONS)}
      <p class='sub'>Each action is gated by the acting role; wrong-actor calls are penalised.</p>
    </div>

    <h2>Reward model (summary)</h2>
    <div class='card'>
      <p>Composable rubric with anti-gaming safeguards. Every step returns a
      <code>reward_components</code> dictionary so training curves are
      interpretable. Closure rewards and SLA penalties are scaled by
      customer-tier multipliers:</p>
      {"".join(f"<span class='pill'>{tier}: x{mult}</span>" for tier, mult in TIER_MULTIPLIER.items())}
    </div>

    <h2>Metadata</h2>
    <div class='card'>
      <pre id='metadata-json'>{metadata_json}</pre>
    </div>
  </div>

  <footer>
    Incident Command Center v{_CONFIG.version} · Built on
    <a href='https://github.com/meta-pytorch/openenv'>OpenEnv</a>.
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
