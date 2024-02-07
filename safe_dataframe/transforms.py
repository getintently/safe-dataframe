"""Base classes for transformations."""

import uuid
from typing import Protocol

import pandas as pd
from loguru import logger


class DataFrameCallable(Protocol):
    """Callable protocol for base functions transforms."""

    def __call__(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """Apply function to data."""
        ...


class BaseTransform:
    """Base class for pandas dataframe transformations."""

    is_transform: bool = True

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identity transform."""
        return data

    @classmethod
    def from_function(
        cls, func: DataFrameCallable, *args, **kwargs
    ) -> "BaseTransform":
        """Generate transform class from functions."""
        name = f"Transform_{func.__name__}_{uuid.uuid4()}"  # type: ignore

        def _call(self: BaseTransform, data: pd.DataFrame) -> pd.DataFrame:
            return func(data, *self.args, **self.kwargs)

        return type(name, (BaseTransform,), {"__call__": _call})(
            *args, **kwargs
        )


class Pipeline(BaseTransform):
    """Pipeline with transformation sequence."""

    verbose: int = 0

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run pipeline while data is no empty."""
        if self.verbose == 1:
            logger.info("Start shape {}", data.shape)
        for func in self.transforms:
            data = func(data)
            if self.verbose == 1:
                logger.info(
                    "{} produces shape {}", func.__class__.__name__, data.shape
                )
            if data.empty:
                break
        return data


class GroupedPipeline(BaseTransform):
    """Pipeline which should be applied after groupby."""

    def __init__(self, group: str, *transforms):
        self.group = group
        if len(transforms) > 1:
            self.transform = Pipeline(*transforms)
        else:
            self.transform = transforms[0]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run pipeline while data is no empty."""
        return data.groupby(self.group).apply(self.transform)


class Concat(BaseTransform):
    """Transform then concatenate."""

    def __init__(self, *transforms, **kwargs):
        self.transforms = transforms
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transforms and concatenate."""
        chunks = []
        for func in self.transforms:
            result = func(data)
            chunks.append(result)
        return pd.concat(chunks, **self.kwargs)
