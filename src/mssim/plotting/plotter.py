import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

def style_ax(ax: plt.Axes, x_col: str, y_label: str, df: pd.DataFrame) -> None:
    if df[x_col].dtype.kind in "iu":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_col.replace("_", " ").capitalize(), fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)


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

        fig, ax = plt.subplots(figsize=(8, 6))

        style = {"linestyle": "-", "marker": "o", "color": None}

        if hue_col and hue_col in df.columns:
            hues = df[hue_col].unique()

            hue_stylings = self.settings.get("hue_stylings", {}).get(str(hue_col), [])
            hue_styling_cfg = {style.pop("value"): style for style in hue_stylings}

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

        fig.tight_layout()

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
