import logging
from pathlib import Path

import pandas as pd


class RatesProcessor:
    """
    A class to process financial rates data.
    It loads price, spot rate, and currency data, computes a final price
    based on recent spot rates, and saves the result.
    """

    def __init__(self, base_path, out_path, log_level=logging.INFO):
        """Initializes paths, and sets up logging."""
        self.base_path = Path(base_path)
        self.out_path = Path(out_path)

        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Loads price, spot, and currency data from files into DataFrames."""
        self.logger.info(f"Loading data from {self.base_path}...")

        self.ccy_df = pd.read_csv(self.base_path / "rates_ccy_data.csv")
        self.price_df = pd.read_parquet(self.base_path / "rates_price_data.parq.gzip")
        self.spot_df = pd.read_parquet(
            self.base_path / "rates_spot_rate_data.parq.gzip"
        )

        # Ensure timestamps are datetimes
        self.price_df["timestamp"] = pd.to_datetime(self.price_df["timestamp"])
        self.spot_df["timestamp"] = pd.to_datetime(self.spot_df["timestamp"])

    def _get_recent_spot_rate(self):
        """For each ccy_pair, find the most recent spot_mid_rate within the previous hour."""
        # Sort for merge_asof
        self.spot_df = self.spot_df.sort_values(["timestamp", "ccy_pair"])
        self.price_df = self.price_df.sort_values(["timestamp", "ccy_pair"])

        # merge_asof will find the last spot rate before price timestamp
        merged = pd.merge_asof(
            self.price_df,
            self.spot_df,
            by="ccy_pair",
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )
        return merged

    def compute_new_prices(self):
        """Calculates the final price based on spot rates and currency conversion rules."""
        self.logger.info("Computing new prices")

        def fill_missing(df, target, others):
            mt = df[target].isna()

            def msg(row):
                missing_cols = [c for c in others if pd.isna(row[c])]
                if not missing_cols:
                    return row[target]
                return "missing " + ", ".join(missing_cols)

            df.loc[mt, target] = df.loc[mt].apply(msg, axis=1)
            return df

        # Get recent spot rate
        merged = self._get_recent_spot_rate()

        # Merge with ccy to get the convert price (bool) and conversion factor
        merged = merged.merge(self.ccy_df, on="ccy_pair", how="left")

        # Set the value of the conversion_factor to 1 if convert_price is False
        mask = merged["convert_price"] == False

        merged.loc[mask, ["conversion_factor", "spot_mid_rate"]] = 1

        # Final price formula
        merged["final_price"] = (merged["price"] / merged["conversion_factor"]) + (
            merged["convert_price"] * merged["spot_mid_rate"]
        )

        # Add message depending on the missing column
        merged = fill_missing(
            merged,
            "final_price",
            ["convert_price", "conversion_factor", "spot_mid_rate"],
        )

        self.result_df = merged.copy()

    def save_output(self):
        """Saves the resulting DataFrame with final prices to a CSV file."""
        self.logger.info("Saving output")

        output_path = self.out_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_df.to_csv(output_path)


if __name__ == "__main__":
    processor = RatesProcessor(base_path="../data", out_path="../results/result.csv")
    processor.load_data()
    processor.compute_new_prices()
    processor.save_output()
