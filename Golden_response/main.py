"""
Main entry point: API server, Streamlit UI, sample data, or full pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config.logging_config import setup_logging
from config.settings import get_settings

logger = setup_logging(__name__)


def run_api(host: str, port: int) -> None:
    """Start FastAPI with uvicorn."""
    import socket

    import uvicorn

    from api.server import create_app

    settings = get_settings()
    settings.ensure_directories()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((host, port)) == 0:
            logger.error(
                "Port %d is already in use. Stop the other API process or use "
                "`python main.py api --port %d`.",
                port,
                port + 1,
            )
            sys.exit(1)

    logger.info("Starting API on %s:%d", host, port)
    uvicorn.run(create_app(), host=host, port=port, reload=False)


def run_streamlit() -> None:
    """Launch Streamlit dashboard."""
    project_root = Path(__file__).parent.resolve()
    app_path = project_root / "app.py"
    logger.info("Starting Streamlit dashboard.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.headless=true",
        ],
        cwd=str(project_root),
        check=False,
    )


def run_sample_data() -> None:
    """Generate sample CSV."""
    from utils.sample_data import generate_sample_data

    settings = get_settings()
    path = settings.data_raw_dir / "sample_sales.csv"
    generate_sample_data(output_path=path)
    logger.info("Sample data created: %s", path)


def run_train_cli(file_path: str) -> None:
    """Train models directly from CLI (bypasses Streamlit)."""
    from models.train import TrainingOrchestrator

    orchestrator = TrainingOrchestrator()
    state = orchestrator.train_all(file_path=file_path)
    logger.info("Training metrics: %s", state.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volatility-Aware AI Sales Forecasting System"
    )
    parser.add_argument(
        "command",
        choices=["api", "ui", "sample-data", "train", "all"],
        help="Command to run",
    )
    parser.add_argument("--host", default=None, help="API host")
    parser.add_argument("--port", type=int, default=None, help="API port")
    parser.add_argument("--file", default=None, help="CSV path for train command")
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    if args.command == "api":
        run_api(host, port)
    elif args.command == "ui":
        run_streamlit()
    elif args.command == "sample-data":
        run_sample_data()
    elif args.command == "train":
        fp = args.file or str(settings.data_raw_dir / "sample_sales.csv")
        run_train_cli(fp)
    elif args.command == "all":
        run_sample_data()
        run_api(host, port)


if __name__ == "__main__":
    main()
