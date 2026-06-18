from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots

plt.style.use(['science','no-latex'])

def style_ax(ax: plt.Axes, x_col: str, y_label: str, df: pd.DataFrame) -> None:
    if df[x_col].dtype.kind in "iu":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_col.replace("_", " ").capitalize(), fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

def _stringify_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_stringify_value(item) for item in value) + "]"
    return str(value)

def _flatten_settings(prefix: str, value: object, excluded_fields: set[str]) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}={_stringify_value(value)}"]

    flattened: list[str] = []
    for key, item in value.items():
        key_str = str(key)
        if key_str in excluded_fields:
            continue

        if isinstance(item, Mapping):
            flattened.extend(_flatten_settings(key_str, item, excluded_fields))
        else:
            flattened.append(f"{key_str}={_stringify_value(item)}")

    return flattened

def _wrap_title(parts: list[str], group_size: int = 4) -> str:
    return "\n".join(
        ", ".join(parts[index : index + group_size])
        for index in range(0, len(parts), group_size)
    ) + "\n"


class MSSIMPlotter:
    """Plotter for MSSIM data based on settings."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings

    def plot(self, df: pd.DataFrame) -> plt.Figure | None:
        """Draw the plot based on the dataframe and settings."""
        if df.empty:
            return None

        axis_cfg = self.settings.get("axis", {})

        x_cfg = axis_cfg.get("x", {})
        y_cfg = axis_cfg.get("y", {})
        hue_cfg = axis_cfg.get("hue", {})

        x_col = x_cfg.get("field_name")
        y_col = y_cfg.get("field_name")
        hue_col = hue_cfg.get("field_name")

        if not x_col or not y_col:
            raise ValueError(
                "Missing required axis configuration: x and y field_names must be defined."
            )

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Missing required columns for plotting: x='{x_col}', y='{y_col}'"
            )

        axis_cfg = self.settings.get("axis", {})
        excluded_fields = {
            str(cfg.get("field_name"))
            for cfg in axis_cfg.values()
            if isinstance(cfg, Mapping) and cfg.get("field_name") is not None
        }

        title_parts = []
        for section_name in ("model", "execution"):
            section = self.settings.get(section_name, {})
            flattened = _flatten_settings(section_name, section, excluded_fields)
            if flattened:
                title_parts.extend(flattened)

        fig, ax = plt.subplots(figsize=(8, 6))

        style = {"linestyle": "-", "marker": "o", "color": None}

        if hue_col and hue_col in df.columns:
            hues = df[hue_col].unique()

            hue_stylings = self.settings.get("hue_stylings", {}).get(str(hue_col), [])
            hue_styling_cfg = {style_def.pop("value"): style_def for style_def in hue_stylings}

            # Sort hues so they appear consistently in legend
            for hue_val in sorted(hues, key=lambda val: str(val)):
                sub_df = df[df[hue_col] == hue_val]

                style = hue_styling_cfg.get(hue_val, {"linestyle": "-", "marker": "o", "color": None})

                self._draw_curve(
                    ax,
                    sub_df,
                    x_col,
                    y_col,
                    label=str(hue_val),
                    style=style,
                )

            ax.legend(title=hue_col.replace("_", " ").capitalize())
        else:
            self._draw_curve(ax, df, x_col, y_col, label=None, style=style)

        # Apply general axis styling from old plotting utilities
        style_ax(ax, x_col, y_col.replace("_", " ").capitalize(), df)

        if title_parts:
            ax.set_title(_wrap_title(title_parts), fontsize=15)

        # fig.tight_layout()

        return fig

    def _draw_curve(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        label: str | None,
        style: dict,
    ) -> None:
        grp = df.groupby(x_col)[y_col]
        xs = grp.mean().index.to_numpy()
        ym = grp.mean().to_numpy()

        if pd.api.types.is_numeric_dtype(df[y_col]):
            ys = grp.std().fillna(0).to_numpy()
        else:
            ys = None
        p = ax.plot(
            xs, ym, label=label, linewidth=2, markersize=7, **style
        )

        if ys is not None:
            fill_color = style.get("color") or p[0].get_color()
            ax.fill_between(xs, ym - ys, ym + ys, color=fill_color, alpha=0.12)
