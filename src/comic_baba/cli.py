"""CLI entry points for the Comic-Baba pipeline."""

import click

from comic_baba import __version__


@click.group()
@click.version_option(__version__, prog_name="comic-baba")
def main() -> None:
    """Comic-Baba — Temporal Hallucinations pipeline."""


@main.command()
@click.option("--config", default="configs/baseline.yaml", show_default=True, help="Config file.")
@click.option("--run-id", default=None, help="Run identifier (auto-generated if omitted).")
def prepare(config: str, run_id: str | None) -> None:
    """Stage 1: validate manifest and extract / standardise frames."""
    from comic_baba.pipelines.prepare_data import run_prepare

    run_prepare(config_path=config, run_id=run_id)


@main.command()
@click.option("--config", default="configs/baseline.yaml", show_default=True, help="Config file.")
@click.option("--run-id", required=True, help="Run identifier from the prepare stage.")
def infer(config: str, run_id: str) -> None:
    """Stage 2: run interpolation (and optional stabilisation)."""
    from comic_baba.pipelines.inference import run_infer

    run_infer(config_path=config, run_id=run_id)


@main.command()
@click.option("--config", default="configs/baseline.yaml", show_default=True, help="Config file.")
@click.option("--run-id", required=True, help="Run identifier from the prepare stage.")
def evaluate(config: str, run_id: str) -> None:
    """Stage 3: compute evaluation metrics."""
    from comic_baba.pipelines.evaluation import run_eval

    run_eval(config_path=config, run_id=run_id)
