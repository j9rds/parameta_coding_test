import logging
from pathlib import Path

import pandas as pd


class StdevProcessor:
    def __init__(self, base_path, out_path, log_level=logging.INFO):
        """Initializes paths, and sets up logging."""
        self.base_path = Path(base_path)
        self.out_path = Path(out_path)

        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Loads the standard dev file into DataFrame"""
        self.logger.info(f"Loading data from {self.base_path}...")
        df = pd.read_parquet(self.base_path / "stdev_price_data.parq.gzip")
        df["snap_time"] = pd.to_datetime(df["snap_time"])
        df = df.sort_values(["security_id", "snap_time"])
        self.df = df.copy()

    def fill_missing_timestamps(self, freq="1h"):
        """
        Ensures each security_id has a complete hourly time index.
        Missing rows will have NaN values for bid/mid/ask.
        """
        self.logger.info(f"Filling missing timestamps with frequency '{freq}'...")
        filled_groups = []
        for sec_id, group in self.df.groupby("security_id"):
            group = group.set_index("snap_time").asfreq(freq)
            group["security_id"] = sec_id
            filled_groups.append(group.reset_index())

        self.df = pd.concat(filled_groups, ignore_index=True)

    def compute_rolling_stdev(self, window=20):
        """Calculates the rolling standard deviation for price columns."""
        self.logger.info(
            f"Computing rolling standard deviation with window size {window}..."
        )
        self.fill_missing_timestamps()
        df = self.df.copy()
        rolled = (
            df.groupby("security_id", group_keys=False)
            .rolling(window=window, min_periods=20, on="snap_time")
            .std(ddof=1)
            .reset_index()
        )
        self.result_df = rolled[
            ["security_id", "snap_time", "bid", "mid", "ask"]
        ].reset_index()

    def save_output(self):
        """Saves the resulting DataFrame to a CSV file."""
        self.logger.info("Saving output")
        output_path = self.out_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_df.to_csv(output_path)


if __name__ == "__main__":
    processor = StdevProcessor(base_path="../data", out_path="../results/result.csv")
    processor.load_data()
    processor.compute_rolling_stdev()
    processor.save_output()
