from pathlib import Path

import pandas as pd
import pytest
from main import RatesProcessor  # <--- CHANGE THIS


@pytest.fixture
def setup_processor(tmp_path):
    """
    Pytest fixture that sets up a temporary testing environment.
    Creates mock CSV/Parquet files, initializes RatesProcessor, and returns it.
    """
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()

    # --- Currency reference data ---
    ccy_df = pd.DataFrame(
        [
            {"ccy_pair": "USDJPY1", "convert_price": False, "conversion_factor": None},
            {"ccy_pair": "USDJPY2", "convert_price": False, "conversion_factor": 100},
            {"ccy_pair": "USDJPY3", "convert_price": False, "conversion_factor": None},
            {"ccy_pair": "EURUSD1", "convert_price": True, "conversion_factor": 100},
            {"ccy_pair": "EURUSD2", "convert_price": True, "conversion_factor": 100},
            {"ccy_pair": "EURUSD3", "convert_price": True, "conversion_factor": None},
            {"ccy_pair": "EURUSD4", "convert_price": None, "conversion_factor": 100},
            {"ccy_pair": "EURUSD5", "convert_price": True, "conversion_factor": None},
        ]
    )

    # --- Price data ---
    price_df = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime("2025-10-16 10:00:00"),
                "ccy_pair": "USDJPY1",
                "price": 150.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 10:00:00"),
                "ccy_pair": "USDJPY2",
                "price": 150.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 10:00:00"),
                "ccy_pair": "USDJPY3",
                "price": 150.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 11:00:00"),
                "ccy_pair": "EURUSD1",
                "price": 11000.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 12:00:00"),
                "ccy_pair": "EURUSD2",
                "price": 12000.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 12:00:00"),
                "ccy_pair": "EURUSD3",
                "price": 12000.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 12:00:00"),
                "ccy_pair": "EURUSD4",
                "price": 12000.0,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 12:00:00"),
                "ccy_pair": "EURUSD5",
                "price": 12000.0,
            },
        ]
    )

    # --- Spot rate data ---
    spot_df = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime("2025-10-16 10:30:00"),
                "ccy_pair": "USDJPY2",
                "spot_mid_rate": 1.1,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 10:30:00"),
                "ccy_pair": "USDJPY3",
                "spot_mid_rate": 1.1,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 10:59:00"),
                "ccy_pair": "EURUSD1",
                "spot_mid_rate": 1.2,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 10:30:00"),
                "ccy_pair": "EURUSD2",
                "spot_mid_rate": 1.1,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 11:59:00"),
                "ccy_pair": "EURUSD3",
                "spot_mid_rate": 1.1,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 11:59:00"),
                "ccy_pair": "EURUSD3",
                "spot_mid_rate": 1.1,
            },
            {
                "timestamp": pd.to_datetime("2025-10-16 11:59:00"),
                "ccy_pair": "EURUSD4",
                "spot_mid_rate": 1.1,
            },
        ]
    )

    # --- Save mock data ---
    ccy_df.to_csv(data_dir / "rates_ccy_data.csv", index=False)
    price_df.to_parquet(data_dir / "rates_price_data.parq.gzip", compression="gzip")
    spot_df.to_parquet(data_dir / "rates_spot_rate_data.parq.gzip", compression="gzip")

    # --- Initialize processor ---
    processor = RatesProcessor(base_path=data_dir, out_path=results_dir / "result.csv")
    return processor


@pytest.mark.parametrize(
    "ccy_pair,timestamp,expected_price",
    [
        # ------------------------------------------------------------------------------
        # If conversion is not required, then the new price is simply the ‘existing price’
        # ------------------------------------------------------------------------------
        (
            "USDJPY1",
            "2025-10-16 10:00:00",
            150.0,
        ),  # No conversion all other params are null
        (
            "USDJPY2",
            "2025-10-16 10:00:00",
            150.0,
        ),  # No conversion all other params have value
        (
            "USDJPY3",
            "2025-10-16 10:00:00",
            150.0,
        ),  # No conversion all some params have value
        # ------------------------------------------------------------------------------
        # If conversion is required, then the new price is: (‘existing price’/ ‘conversion factor’) + ‘spot_mid_rate’
        # ------------------------------------------------------------------------------
        ("EURUSD1", "2025-10-16 11:00:00", pytest.approx(111.2)),  # Valid conversion
        # ------------------------------------------------------------------------------
        # If there is insufficient data to create a new price then capture this fact in some way
        # ------------------------------------------------------------------------------
        ("EURUSD2", "2025-10-16 12:00:00", "missing spot_mid_rate"),  # Old spot rate
        (
            "EURUSD3",
            "2025-10-16 12:00:00",
            "missing conversion_factor",
        ),  # Missing conversion factor
        (
            "EURUSD4",
            "2025-10-16 12:00:00",
            "missing convert_price",
        ),  # Missing convert price boolean
        (
            "EURUSD5",
            "2025-10-16 12:00:00",
            "missing conversion_factor, spot_mid_rate",
        ),  # Missing multiple paramters
    ],
)
def test_final_prices(setup_processor, ccy_pair, timestamp, expected_price):
    """
    Parametrized tests for all conversion/no-conversion/invalid-data cases,
    including null conversion_factor when convert_price=False.
    """
    processor = setup_processor
    processor.load_data()
    processor.compute_new_prices()
    result_df = processor.result_df

    test_row = result_df[
        (result_df["ccy_pair"] == ccy_pair)
        & (result_df["timestamp"] == pd.to_datetime(timestamp))
    ]

    assert not test_row.empty
    assert test_row["final_price"].iloc[0] == expected_price
