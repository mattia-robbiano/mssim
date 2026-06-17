import argparse
import logging
from datetime import datetime
from pathlib import Path

from mssim.config import PROJECT_ROOT
from mssim.plotting.parser import MSSIMPlotParser
from mssim.plotting.postprocessor import MSSIMPlotPostProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Parse and post-process MSSIM plotting data."
    )
    parser.add_argument(
        "settings", type=str, help="Path to the plot settings JSON file."
    )
    args = parser.parse_args()

    settings_path = Path(args.settings)

    if not settings_path.exists():
        logging.error(f"Settings file not found: {settings_path}")
        return
    logging.info(f"Loading settings from: {settings_path}")
    plot_parser = MSSIMPlotParser(settings_path)

    logging.info("Parsing input data...")
    raw_df = plot_parser.parse()

    if raw_df is None or raw_df.empty:
        logging.warning("No data found or parsed dataframe is empty.")
        return

    logging.info(f"Raw data shape: {raw_df.shape}")

    post_processor = MSSIMPlotPostProcessor(plot_parser)

    logging.info("Filtering and processing data...")
    filtered_df = post_processor.process(raw_df)
    logging.info(f"Filtered data shape: {filtered_df.shape}")

    # Generate the output path for the processed data
    now = datetime.now()

    data_output_cfg = plot_parser.settings.get("output", {}).get("data", {})

    template = data_output_cfg.get("filename", "{date}_{time}")
    rel_path = data_output_cfg.get("rel_path", "")
    fmt = data_output_cfg.get("format", "csv")

    filename = template.format(date=now.strftime("%Y%m%d"), time=now.strftime("%H-%M"))

    out_dir = PROJECT_ROOT / rel_path
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{filename}.{fmt}"

    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()
