"""Base class for data."""

import abc
import warnings
from functools import cached_property
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import pandera as pa
from pydantic import BaseModel, computed_field

from safe_dataframe.transforms import BaseTransform

ColumnsType = TypeVar("ColumnsType", bound="Columns")
BaseDataType = TypeVar("BaseDataType", bound="BaseData")


class Columns(BaseModel):
    """Container for column names and pandera schema."""

    @abc.abstractmethod
    def get_data_schema(self) -> pa.DataFrameSchema:
        """Calculate data schema."""

    @computed_field(alias="DataFrameSchema", repr=False)  # type: ignore[misc]
    @cached_property
    def data_schema(self) -> pa.DataFrameSchema:
        """Return data schema."""
        return self.get_data_schema()

    @property
    def types(self) -> dict[str, pa.DataType]:
        """Get types."""
        return self.data_schema.dtypes

    def dump_column_names(self) -> dict:
        """Dump only column names without schema."""
        return self.model_dump(exclude={"data_schema"})

    def set_prefix(self: ColumnsType, prefix: str) -> ColumnsType:
        """Construct with prefix."""
        columns = self.dump_column_names()
        new_attrs = {name: prefix + val for name, val in columns.items()}
        return self.model_validate(new_attrs)

    def validate_data(self, data: pd.DataFrame):
        """Run validation."""
        self.data_schema.validate(data)

    def columns(self) -> list[str]:
        """Get columns attributes."""
        return list(self.dump_column_names().keys())

    def get_names(self) -> list[str]:
        """Get string names of columns."""
        return list(self.dump_column_names().values())

    def dump_dict(self) -> dict:
        """Dump model as dictionary."""
        json_dict = self.dump_column_names()
        json_dict["data_schema"] = self.data_schema.to_json()
        return json_dict

    def intersection(self, names: list[str]) -> list[str]:
        """Return names intersection."""
        return list(set(self.get_names()) & set(names))


class BaseData(Generic[ColumnsType], metaclass=abc.ABCMeta):
    """Base class for data."""

    _column_class: type[ColumnsType]

    def __init__(
        self,
        data: pd.DataFrame,
        columns: ColumnsType | None = None,
        skip_check: bool = False,  # noqa: FBT002
    ):
        if columns is None:
            columns = self._column_class()
        self.columns = columns
        if not skip_check:
            self.columns.validate_data(data)
        else:
            warnings.warn(
                f"Data check skipped {self.__class__.__name__}.",
                UserWarning,
                stacklevel=1,
            )
        self._data = data

    def prefix_columns(self: BaseDataType, prefix: str) -> BaseDataType:
        """Get data with prefix in column names."""
        new = self.data.rename(
            columns={name: prefix + name for name in self.data.columns}
        )
        new_columns = self.columns.set_prefix(prefix)
        return self.from_dataframe(new, columns=new_columns)

    def new_data(self: BaseDataType, data: pd.DataFrame) -> BaseDataType:
        """Return entity with new data."""
        return self.from_dataframe(data, columns=self.columns)

    @classmethod
    def default_columns(cls: type[BaseDataType]) -> ColumnsType:
        """Get default column schema."""
        return cls._column_class()

    @property
    def c(self) -> ColumnsType:
        """Shortcut for columns."""
        return self.columns

    @property
    def data(self) -> pd.DataFrame:
        """Get data."""
        return self._data

    @classmethod
    def from_dataframe(
        cls: type[BaseDataType],
        data: pd.DataFrame,
        columns: ColumnsType | None = None,
        transform: BaseTransform | None = None,
        **kwargs,
    ) -> BaseDataType:
        """Construct object from dataframe and transform."""
        if transform is not None:
            data = transform(data)
        if columns is None:
            columns = cls.default_columns()
        return cls(data, columns=columns, **kwargs)

    def transform(
        self: BaseDataType, transform: BaseTransform
    ) -> BaseDataType:
        """Transform data."""
        return self.from_dataframe(
            self.data, transform=transform, columns=self.columns
        )

    def get_values_presence(self, columns: list[str]) -> pd.Series:
        """Get data presence for columns."""
        return self.data[columns].notna().sum(axis=0) / len(self.data.index)

    def unique(self, column: str) -> np.ndarray:
        """Get unique values."""
        return self.data[column].unique()

    def truncate_columns(self: BaseDataType) -> BaseDataType:
        """Remove redundant columns."""
        return self.new_data(self.data[self.c.get_names()])


class BaseDataTransform(BaseTransform, Generic[ColumnsType]):
    """Data transformations interface.

    We want to decouple transforms from actual data
    to be reused in composite data objects.
    """

    def __init__(self, columns: ColumnsType, *args, **kwargs):
        self._columns = columns
        super().__init__(*args, **kwargs)

    @property
    def c(self) -> ColumnsType:
        """Get columns shortcut."""
        return self._columns

    @property
    def columns(self) -> ColumnsType:
        """Get columns."""
        return self._columns


class BaseDataTransformContainer(Generic[ColumnsType]):
    """Class contains transforms related to data."""

    def __init__(self, columns: ColumnsType):
        self.columns = columns

    def __getattribute__(self, __name: str) -> Any:  # noqa: ANN401
        """Put columns in transform constructor when getter."""
        attr = object.__getattribute__(self, __name)

        if getattr(attr, "is_transform", False):

            def transform(*args, **kwargs) -> BaseTransform:
                """Transform constructor decorator."""
                return attr(*args, columns=self.columns, **kwargs)

            return transform
        return attr


LeftColumnsType = TypeVar("LeftColumnsType", bound="Columns")
RightColumnsType = TypeVar("RightColumnsType", bound="Columns")


class MergeContainer(Generic[LeftColumnsType, RightColumnsType]):
    """Simple container for merged data."""

    def __init__(
        self,
        data: pd.DataFrame,
        left: LeftColumnsType,
        right: RightColumnsType,
    ):
        self._data = data
        self.left = left
        self.right = right

    @property
    def data(self) -> pd.DataFrame:
        """Get data."""
        return self._data

    @property
    def l(self) -> LeftColumnsType:  # noqa: E743
        """Left columns."""
        return self.left

    @property
    def r(self) -> RightColumnsType:
        """Left columns."""
        return self.right

    @staticmethod
    def _prefix_columns(
        left: BaseData, right: BaseData, prefix: tuple[str, str]
    ) -> tuple[BaseData, BaseData]:
        if len(prefix[0]) > 0:
            left = left.prefix_columns(prefix[0])
        if len(prefix[1]) > 0:
            right = right.prefix_columns(prefix[1])
        return left, right
