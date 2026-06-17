import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mssim.config import PROJECT_ROOT
from mssim.plotting.plotter import MSSIMPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Draw a graph from a CSV using MSSIM plot settings."
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "settings", type=str, help="Path to the plot settings JSON file."
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    settings_path = Path(args.settings)

    if not csv_path.exists():
        logging.error(f"CSV file not found: {csv_path}")
        return

    if not settings_path.exists():
        logging.error(f"Settings file not found: {settings_path}")
        return

    logging.info(f"Loading settings from: {settings_path}")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    logging.info(f"Reading CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty:
        logging.warning("Input CSV is empty. Nothing to plot.")
        return

    logging.info("Initializing plotter...")
    plotter = MSSIMPlotter(settings)

    logging.info("Drawing plot...")
    fig = plotter.plot(df)

    if fig is None:
        logging.warning("Plotting failed or dataframe was empty.")
        return

    # Generate the output path for the processed data
    now = datetime.now()

    plot_output_cfg = settings.get("output", {}).get("plot", {})

    template = plot_output_cfg.get("filename", "{date}_{time}")
    rel_path = plot_output_cfg.get("rel_path", "")
    fmt = plot_output_cfg.get("format", "png")

    filename = template.format(date=now.strftime("%Y%m%d"), time=now.strftime("%H-%M"))

    out_dir = PROJECT_ROOT / rel_path
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{filename}.{fmt}"

    fig.savefig(output_path, dpi=300, bbox_inches="tight", format=fmt)

    plt.close(fig)

    logging.info(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
