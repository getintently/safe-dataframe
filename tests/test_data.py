"""Test base data classes."""

import pandas as pd
import pandera as pa
import pytest

from safe_dataframe.data import BaseData, Columns
from safe_dataframe.transforms import BaseTransform, Pipeline


def test_columns_names() -> None:
    """Test column names function."""

    class TestColumns(Columns):
        """Test columns class."""

        column_a: str = "Name A"
        column_b: str = "Name B"

        def get_data_schema(self) -> pa.DataFrameSchema:
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(str, nullable=False),
                    self.column_b: pa.Column(int, nullable=False),
                }
            )

    columns = TestColumns()

    assert {"Name A", "Name B"} == set(columns.get_names())


def test_safe_dataframe() -> None:
    """Test safe dataframe works."""

    class TestColumns(Columns):
        """Test columns class."""

        column_a: str = "Name A"
        column_b: str = "Name B"

        def get_data_schema(self) -> pa.DataFrameSchema:
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(str, nullable=False),
                    self.column_b: pa.Column(int, nullable=False),
                }
            )

    class TestData(BaseData[TestColumns]):
        """Test dataframe wrapper."""

        _column_class = TestColumns

    columns = TestColumns()
    data = pd.DataFrame(
        {
            "Name A": ["a", "b", "c"],
            "Name B": [1, 2, 3],
        }
    )

    test_data = TestData(data=data, columns=columns)
    assert test_data.data.shape == (3, 2)


def test_safe_dataframe_fail() -> None:
    """Test safe dataframe works."""

    class TestColumns(Columns):
        """Test columns class."""

        column_a: str = "Name A"
        column_b: str = "Name B"

        def get_data_schema(self) -> pa.DataFrameSchema:
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(str, nullable=False),
                    self.column_b: pa.Column(int, nullable=False),
                }
            )

    class TestData(BaseData[TestColumns]):
        """Test dataframe wrapper."""

        _column_class = TestColumns

    columns = TestColumns()
    data = pd.DataFrame(
        {
            "Name A": ["a", "b", "c"],
            "Name B": [1, 2, pd.NA],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        TestData(data=data, columns=columns)


def test_safe_dataframe_transform() -> None:
    """Test safe dataframe works."""

    class TestColumns(Columns):
        """Test columns class."""

        column_a: str = "Name A"
        column_b: str = "Name B"

        def get_data_schema(self) -> pa.DataFrameSchema:
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(str, nullable=False),
                    self.column_b: pa.Column(int, nullable=False),
                }
            )

    class TestData(BaseData[TestColumns]):
        """Test dataframe wrapper."""

        _column_class = TestColumns

    columns = TestColumns()
    data = pd.DataFrame(
        {
            "Name A": ["a", "b", "c"],
            "Name B": [1, 2, pd.NA],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        TestData(data=data, columns=columns)

    class FillNaTransform(BaseTransform):
        """Test transform filling NaNs."""

        def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.fillna(0)

    test_data = TestData.from_dataframe(
        data, columns=columns, transform=Pipeline(FillNaTransform())
    )
    assert test_data.data.shape == (3, 2)
