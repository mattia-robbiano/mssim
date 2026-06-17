import pandas as pd

from mssim.plotting.parser import MSSIMPlotParser


class MSSIMPlotPostProcessor:
    """Post-processor to filter and clean parsed MSSIM data based on settings."""

    def __init__(self, parser: MSSIMPlotParser) -> None:
        self.parser = parser
        self.settings = parser.settings

    def _get_config_keys_and_values(
        self,
    ) -> tuple[set[str], dict[str, any], dict[str, list]]:
        """Extract allowed columns, static filters, and dynamic axis exclusions from config."""
        static_filters = {}
        exclude_map = {}
        allowed_cols = set()
        dynamic_fields = set()

        axis_cfg = self.settings.get("axis", {})
        for _, axis_data in axis_cfg.items():
            field_name = axis_data.get("field_name")

            dynamic_fields.add(field_name)
            exclude_map[field_name] = axis_data.get("exclude_values", [])

        # Extract allowed keys and static expected values from model config
        model_cfg = self.settings.get("model", {})
        kwargs = model_cfg.pop("kwargs", {})

        flat_model_cfg = {**model_cfg, **kwargs}
        for k, v in flat_model_cfg.items():
            allowed_cols.add(k)
            if k not in dynamic_fields:
                static_filters[k] = v

        # Extract allowed keys and static expected values from execution config
        exec_cfg = self.settings.get("execution", {})
        for k, v in exec_cfg.items():
            allowed_cols.add(k)
            if k not in dynamic_fields:
                static_filters[k] = v

        # Combine allowed columns with dynamic fields to ensure they are retained in the dataframe
        allowed_cols = allowed_cols.union(dynamic_fields)

        return allowed_cols, static_filters, exclude_map

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the dataframe according to the configuration settings."""
        if df.empty:
            return df

        allowed_cols, static_filters, exclude_map = self._get_config_keys_and_values()

        # Remove columns that are not in allowed_cols
        keep_cols = [c for c in df.columns if c in allowed_cols]
        df = df[keep_cols]

        # Remove rows where values of static fields do not match config values
        for col, val in static_filters.items():
            if val is None:
                df = df[df[col].isna()]
            else:
                df = df[df[col] == val]

        # Exclude values on dynamic axes if specified
        for col, excludes in exclude_map.items():
            df = df[~df[col].isin(excludes)]

        # Delete duplicate rows if any
        df = df.drop_duplicates()

        return df
