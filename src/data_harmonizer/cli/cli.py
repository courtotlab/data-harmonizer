import os
from pathlib import Path
import typer
from rich.table import Table
from rich.console import Console
from typing_extensions import Annotated
from data_harmonizer.data.schema_data import get_schema_features
from data_harmonizer.data.synthetic_data import generate
from data_harmonizer.data.split_data import run_split


app = typer.Typer(
    name="data-harmonizer",
    help="Data Harmonizer CLI",
    no_args_is_help=True,
    epilog="Copyright (c) 2025 OICR",
)


@app.command()
def train(
    target: Annotated[
        Path, typer.Argument(help="Path to the target LinkML schema file.")
    ],
    outdir: Annotated[
        Path, typer.Option("-o", "--outdir", help="Output directory for the training results.")
    ] = Path("./"),
):
    # Check if the target file exists
    if not target.exists():
        typer.echo(f"Error: The target file '{target}' does not exist.")
    if not outdir.exists():
        if typer.confirm("Output directory does not exist. Create it?", abort=True):
            outdir.mkdir(parents=True, exist_ok=True)

    schema_df = get_schema_features(target)
    synthetic_df = generate(schema_df)
    triplet_dfs = run_split(schema_df, synthetic_df)

    # Save the triplet dataframes to CSV files
    split_dir = outdir / f"{target.stem}_synth"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for data_type in triplet_dfs.keys():
        split_path = split_dir / f"{data_type}.csv"
        triplet_dfs[data_type].to_csv(split_path, index=False)
        split_paths[data_type] = split_path

    # set the output paths
    log_path = outdir / f"{target.stem}_logs"
    model_path = outdir / f"{target.stem}_model.ckpt"

    # run training
    from data_harmonizer.modeling.train import train as model_train
    model_train(split_paths, log_path, model_path)
    typer.echo(
        f"Training completed. Model saved to {model_path} and logs to {log_path}."
    )


@app.command()
def predict(
    target: Annotated[
        Path, typer.Argument(help="Path to the target LinkML schema file.")
    ],
    source: Annotated[
        Path, typer.Argument(help="Path to the source LinkML schema file.")
    ],
    model: Annotated[
        Path, typer.Argument(help="Path to the trained model file.")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output file for the predictions.")
    ] = Path("./assignments.csv")
):
    # Check if the source file exists
    if not target.exists():
        typer.echo(f"Error: The target file '{target}' does not exist.")
        raise typer.Exit(code=os.EX_NOINPUT)
    if not source.exists():
        typer.echo(f"Error: The source file '{source}' does not exist.")
        raise typer.Exit(code=os.EX_NOINPUT)
    if not model.exists():
        typer.echo(f"Error: The model file '{model}' does not exist.")
        raise typer.Exit(code=os.EX_NOINPUT)

    from data_harmonizer.main import predict as md_predict
    assignment_df = md_predict(
        target=get_schema_features(target),
        source=get_schema_features(source),
        model_path=str(model)
    )

    console = Console()
    table = Table("Target Field", "Source Field", "Cost")
    
    for _, row in assignment_df.iterrows():
      table.add_row(str(row.iloc[0]), str(row.iloc[1]), f"{row.iloc[2]:.4f}")

    console.print(table)

    assignment_df.to_csv(output, index=False)
    typer.echo(f"Assignments saved to {output}.")

if __name__ == "__main__":
    app()
