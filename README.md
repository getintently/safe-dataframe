# safe-data

Data framework to fix data schema for pandas dataframes.

This framework uses pandera checks.

General usage:

```python
from safe_dataframe import BaseData, Columns, Pipeline

class WeightDataColumns(Columns):
    """Signal NFX profile data columns."""

    id_: str = "id"
    weight: str = "Weight Column"

    def get_data_schema(self) -> pa.DataFrameSchema:
        """Get data schema."""
        return pa.DataFrameSchema(
            {
                self.id_: pa.Column(str, unique=True, nullable=False),
                self.weight: pa.Column(float, unique=False, nullable=True),
            },
            coerce=True,
        )


class WeightData(BaseData[WeightDataColumns], DataRawBase):

    _column_class = WeightDataColumns


import pandas as pd

raw = pd.DataFrame({"id": [1, 2, 3], "Weight Column": [10, 20, 30]})
data = WeightData(raw)
```
