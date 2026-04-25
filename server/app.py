from openenv.core.env_server import create_fastapi_app
from models import SREAction, SREObservation
from server.environment import SREEnvironment
from fastapi.responses import HTMLResponse
import uvicorn

dashboard_content = r"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>SRE Debugging | OpenEnv Dashboard</title>
    <style>
        :root { --primary: #3b82f6; --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; }
        body { font-family: -apple-system, sans-serif; background-color: var(--bg); color: var(--text); padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; background: var(--card); padding: 2rem; border-radius: 1rem; }
        code { background: #334155; padding: 0.2rem; border-radius: 0.25rem; font-family: monospace; color: #38bdf8; }
    </style>
</head>
<body>
    <div class='container'>
        <h1>Long-Horizon SRE Debugging</h1>
        <p>Theme 3.1: Professional Tasks - Multi-step agent triage simulator.</p>
        
        <h2>Action Space</h2>
        <ul>
            <li><code>query_logs(service_name)</code></li>
            <li><code>check_metrics(dashboard_id)</code></li>
            <li><code>resolve_ticket(root_cause)</code></li>
        </ul>
        
        <h2>Reward Logic</h2>
        <p>Correctly resolving a ticket yields <b>+1.0</b>. Querying logs or checking metrics costs a slight penalty of <b>-0.1</b> to encourage agent efficiency. Wrong resolution gives <b>-1.0</b>.</p>
    </div>
</body>
</html>
"""

app = create_fastapi_app(SREEnvironment, SREAction, SREObservation)

@app.get('/', response_class=HTMLResponse)
@app.get('/web', response_class=HTMLResponse)
async def root():
    return dashboard_content

def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
