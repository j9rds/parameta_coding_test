from pathlib import Path

import numpy as np
import pandas as pd
import pytest
# Import the class from your script
from main import StdevProcessor
from pandas.testing import assert_frame_equal


@pytest.fixture
def temp_dir(tmp_path):
    """
    A pytest fixture to create temporary input/output directories for tests,
    ensuring a clean state for each test run.
    """
    base_path = tmp_path / "input"
    out_path = tmp_path / "output"
    base_path.mkdir()
    return base_path, out_path


def create_parquet_file(df, path):
    """Helper function to create a test parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="gzip")


# --- Test Cases ---


def test_happy_path_sufficient_data(temp_dir):
    """
    Tests the standard deviation calculation for a single security with enough
    contiguous data (21 points) for two calculations.
    """
    # Set up paths and create input data
    base_path, out_path = temp_dir
    input_file_path = base_path / "stdev_price_data.parq.gzip"

    start_time = "2021-11-20 00:00:00"
    timestamps = pd.to_datetime(pd.date_range(start=start_time, periods=21, freq="h"))
    data = {
        "snap_time": timestamps,
        "security_id": [101] * 21,
        "bid": range(21),
        "mid": [v + 10 for v in range(21)],
        "ask": [v + 20 for v in range(21)],
    }
    input_df = pd.DataFrame(data)
    create_parquet_file(input_df, input_file_path)

    # Run the processor
    processor = StdevProcessor(base_path, out_path / "output.csv")
    processor.load_data()
    processor.compute_rolling_stdev(window=20)
    result_df = processor.result_df

    # Verify the results
    # The first calculation happens at the 20th data point (index 19)
    expected_std_1 = np.std(range(20), ddof=1)  # Std dev for values 0-19
    expected_std_2 = np.std(range(1, 21), ddof=1)  # Std dev for values 1-20

    # First 19 values for the std dev column should be NaN
    assert result_df["bid_std"].iloc[:19].isnull().all()
    # Check the calculated value at index 19
    assert np.isclose(result_df["bid_std"].iloc[19], expected_std_1)
    # Check the calculated value at index 20
    assert np.isclose(result_df["bid_std"].iloc[20], expected_std_2)


def test_insufficient_data(temp_dir):
    """
    Tests that no standard deviation is calculated if there are fewer than
    the minimum required periods (19 points vs 20 needed).
    """
    #
    base_path, out_path = temp_dir
    input_file_path = base_path / "stdev_price_data.parq.gzip"

    start_time = "2021-11-20 00:00:00"
    timestamps = pd.to_datetime(pd.date_range(start=start_time, periods=19, freq="h"))
    input_df = pd.DataFrame(
        {
            "snap_time": timestamps,
            "security_id": [101] * 19,
            "bid": range(19),
            "mid": range(19),
            "ask": range(19),
        }
    )
    create_parquet_file(input_df, input_file_path)

    #
    processor = StdevProcessor(base_path, out_path / "output.csv")
    processor.load_data()
    processor.compute_rolling_stdev(window=20)
    result_df = processor.result_df

    # All standard deviation columns should be NaN
    assert result_df[["bid_std", "mid_std", "ask_std"]].isnull().all().all()


def test_gap_in_data_and_ffill(temp_dir):
    """
    Tests how the processor handles a gap in timestamps. It should fill the
    gap with NaN and then forward-fill the last valid std dev calculation.
    """
    # Create data with a missing hour at 04:00
    base_path, out_path = temp_dir
    input_file_path = base_path / "stdev_price_data.parq.gzip"

    start_time = "2021-11-20 00:00:00"
    period = 51
    timestamps = pd.to_datetime(
        pd.date_range(start=start_time, periods=period + 1, freq="h")
    )
    # Drop the timestamp for the 30th hour
    timestamps = timestamps.drop(timestamps[30])

    # Generate a more realistic random price series (a random walk)
    bids = 100 + np.random.randn(period).cumsum()
    # Create mid and ask prices with a small, random spread
    mids = bids + np.random.uniform(0.01, 0.05, size=period)
    asks = mids + np.random.uniform(0.01, 0.05, size=period)
    input_df = pd.DataFrame(
        {
            "snap_time": timestamps,
            "security_id": [101] * period,
            "bid": bids,
            "mid": mids,
            "ask": asks,
        }
    )
    create_parquet_file(input_df, input_file_path)

    #
    processor = StdevProcessor(base_path, out_path / "output.csv")
    processor.load_data()
    processor.compute_rolling_stdev(window=20)
    result_df = processor.result_df

    # A valid std dev is calculated at index 19
    valid_std_time = pd.Timestamp("2021-11-20 19:00:00")
    first_valid_std = result_df.loc[
        result_df.snap_time == valid_std_time, "bid_std"
    ].iloc[0]
    assert np.isclose(first_valid_std, result_df["bid"][:20].std())

    # No null values after index 19
    assert result_df["bid_std"].iloc[19:].notna().all()

    # The values of std of index 30-49 should be equal to the std of index 29
    assert (result_df["bid_std"] == result_df["bid_std"].iloc[29]).any()
