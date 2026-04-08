"""
FastAPI application for HARvestGym.

Exposes HARvestGymEnvironment over HTTP endpoints compatible with OpenEnv EnvClient.

Endpoints:
    POST /reset  — Reset the environment
    POST /step   — Execute an action
    GET  /state  — Get current state
    GET  /schema — Get action/observation schemas
    GET  /health — Health check
    WS   /ws     — WebSocket for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from .models import HarvestGymAction, HarvestGymObservation, HARvestGymEnvironment
except ModuleNotFoundError:
    from server.models import HarvestGymAction, HarvestGymObservation, HARvestGymEnvironment

app = create_app(
    HARvestGymEnvironment,
    HarvestGymAction,
    HarvestGymObservation,
    env_name="HARvestGym",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port != 8000:
        main(port=args.port)
    else:
        main()
