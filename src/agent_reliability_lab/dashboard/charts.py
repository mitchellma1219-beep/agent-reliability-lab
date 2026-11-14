"""Plotly figures for the dashboard.

Design rules applied here:

* the form follows the job: single numbers are stat tiles (not charts), nominal categories
  are bars in one hue, history is a line, two measures never share one axis (each gets its
  own small chart);
* categorical colour is used only where series must be told apart (baseline vs candidate,
  kinds of trace step), in a fixed slot order: blue, orange, aqua. These three were checked
  with a palette validator for colour-vision-deficiency separation in light and dark mode;
* status colours mean pass / fail only and always come with a text label;
* thin bars with rounded data ends, 2 px lines, hairline solid grid, text in ink colours;
* every chart has a table view next to it in the app, so no value is hover-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class Palette:
    series: tuple[str, str, str]
    good: str
    critical: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    surface: str


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    good="#0ca30c",
    critical="#d03b3b",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    surface="#ffffff",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70"),
    good="#0ca30c",
    critical="#d03b3b",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    surface="#0e1117",
)

STEP_KINDS = ("model call", "tool call", "wait")
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(theme: str | None) -> Palette:
    return DARK if theme == "dark" else LIGHT


def _layout(fig: go.Figure, colours: Palette, height: int, legend: bool) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 24, "t": 8, "b": 8},
        font={"family": _FONT, "size": 13, "color": colours.secondary},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": colours.secondary},
        },
        hoverlabel={"font": {"family": _FONT, "size": 13}},
        bargap=0.5,
    )
    axis = {
        "gridcolor": colours.grid,
        "gridwidth": 1,
        "linecolor": colours.axis,
        "zeroline": False,
        "tickfont": {"color": colours.muted},
        "title": {"font": {"color": colours.muted}},
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _bar_height(rows: int) -> int:
    return max(140, 34 * rows + 56)


def rate_bars(frame: pd.DataFrame, label: str, value: str, theme: str | None = None) -> go.Figure:
    """Horizontal bars of a rate (0..1) per nominal category, one hue, value at the tip."""
    colours = palette(theme)
    data = frame.dropna(subset=[value]).iloc[::-1]
    counted = {"passed", "cases"} <= set(data.columns)
    fig = go.Figure(
        go.Bar(
            x=data[value],
            y=data[label],
            orientation="h",
            marker={"color": colours.series[0], "cornerradius": 4},
            text=[f"{item:.0%}" for item in data[value]],
            textposition="outside",
            textfont={"color": colours.secondary},
            cliponaxis=False,
            customdata=data[["passed", "cases"]] if counted else None,
            hovertemplate=(
                "<b>%{x:.1%}</b> (%{customdata[0]} of %{customdata[1]} cases)"
                "<br>%{y}<extra></extra>"
                if counted
                else "<b>%{x:.1%}</b><br>%{y}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, colours, _bar_height(len(data)), legend=False)


def count_bars(frame: pd.DataFrame, label: str, value: str, theme: str | None = None) -> go.Figure:
    """Ranked horizontal bars of counts, one hue, value at the tip."""
    colours = palette(theme)
    data = frame.sort_values(value, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data[value],
            y=data[label],
            orientation="h",
            marker={"color": colours.series[0], "cornerradius": 4},
            text=[f"{item:,}" for item in data[value]],
            textposition="outside",
            textfont={"color": colours.secondary},
            cliponaxis=False,
            hovertemplate="<b>%{x:,}</b><br>%{y}<extra></extra>",
        )
    )
    top = float(data[value].max()) if len(data) else 1.0
    fig.update_xaxes(range=[0, top * 1.15 or 1.0], tickformat=",d")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, colours, _bar_height(len(data)), legend=False)


def trend_line(
    frame: pd.DataFrame,
    value: str,
    theme: str | None = None,
    *,
    as_rate: bool = False,
    unit: str = "",
    decimals: int = 0,
) -> go.Figure:
    """One measure over the versions of one agent: a single 2 px line, last point labelled."""
    colours = palette(theme)
    data = frame.dropna(subset=[value])
    values = [float(item) for item in data[value]]
    fmt = ".1%" if as_rate else f",.{decimals}f"
    last = len(values) - 1
    labels = [""] * len(values)
    if values:
        labels[last] = f"{values[last]:.0%}" if as_rate else f"{values[last]:,.{decimals}f}{unit}"
    falling = last > 0 and values[last] < values[last - 1]
    fig = go.Figure(
        go.Scatter(
            x=data["version"],
            y=values,
            mode="lines+markers+text",
            line={"color": colours.series[0], "width": 2, "shape": "linear"},
            marker={
                "color": colours.series[0],
                "size": 9,
                "line": {"width": 2, "color": colours.surface},
            },
            text=labels,
            textposition="bottom center" if falling else "top center",
            textfont={"color": colours.secondary},
            cliponaxis=False,
            customdata=data[["created"]],
            hovertemplate=f"<b>%{{y:{fmt}}}{unit}</b><br>version %{{x}} (%{{customdata[0]}})"
            "<extra></extra>",
        )
    )
    fig.update_xaxes(type="category", showgrid=False, title_text="agent version")
    if as_rate:
        fig.update_yaxes(range=[0, 1.1], tickformat=".0%", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    else:
        top = max(values, default=1.0) or 1.0
        fig.update_yaxes(range=[0, top * 1.2], tickformat=f",.{decimals}f", nticks=5)
    fig.update_layout(hovermode="x")
    figure = _layout(fig, colours, 250, legend=False)
    figure.update_layout(margin={"l": 8, "r": 64, "t": 16, "b": 8})
    return figure


def comparison_bars(frame: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """Baseline against candidate pass rate per category: two series, fixed colour slots."""
    colours = palette(theme)
    data = frame.iloc[::-1]
    fig = go.Figure()
    for slot, column in enumerate(("candidate", "baseline")):
        fig.add_trace(
            go.Bar(
                x=data[column],
                y=data["category"],
                orientation="h",
                name=column,
                marker={"color": colours.series[1 - slot], "cornerradius": 4},
                legendrank=2 - slot,
                text=[f"{item:.0%}" for item in data[column]],
                textposition="outside",
                textfont={"color": colours.secondary},
                cliponaxis=False,
                hovertemplate=f"<b>%{{x:.1%}}</b> {column}<br>%{{y}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargroupgap=0.25)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    fig.update_yaxes(showgrid=False)
    figure = _layout(fig, colours, max(180, 52 * len(data) + 70), legend=True)
    figure.update_layout(bargap=0.35)
    return figure


def timeline(steps: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """Accounted time per step: one bar per step, coloured by kind, failures labelled."""
    colours = palette(theme)
    data = steps.iloc[::-1]
    fig = go.Figure()
    labels = [
        f"{int(row.step)}  {row.name_}"
        for row in data.rename(columns={"name": "name_"}).itertuples()
    ]
    for slot, kind in enumerate(STEP_KINDS):
        mask = (data["kind"] == kind).tolist()
        if not any(mask):
            continue
        rows = data[mask]
        chosen = [label for label, keep in zip(labels, mask, strict=True) if keep]
        failed = [str(outcome).startswith("failed") for outcome in rows["outcome"]]
        fig.add_trace(
            go.Bar(
                x=rows["duration (ms)"],
                base=rows["start (ms)"],
                y=chosen,
                orientation="h",
                name=kind,
                marker={"color": colours.series[slot], "cornerradius": 3},
                text=[
                    f"  ✕ {outcome}" if bad else ""
                    for outcome, bad in zip(rows["outcome"], failed, strict=True)
                ],
                textposition="outside",
                textfont={"color": colours.critical},
                cliponaxis=False,
                customdata=rows[["outcome", "detail"]],
                hovertemplate="<b>%{x:,.0f} ms</b> from %{base:,.0f} ms<br>%{y}: %{customdata[0]}"
                "<br>%{customdata[1]}<extra></extra>",
            )
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="accounted time (ms)", rangemode="tozero", tickformat=",d")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=labels)
    return _layout(fig, colours, max(160, 30 * len(data) + 90), legend=True)


def figure_summary(fig: go.Figure) -> dict[str, Any]:
    """A small description of a figure, used by tests to check what was drawn."""
    return {
        "traces": len(fig.data),
        "types": sorted({trace.type for trace in fig.data}),
        "names": [trace.name for trace in fig.data if trace.name],
        "colours": [
            trace.marker.color if isinstance(trace.marker.color, str) else None
            for trace in fig.data
        ],
        "legend": bool(fig.layout.showlegend),
        "y_axes": sum(1 for key in fig.layout.to_plotly_json() if key.startswith("yaxis")),
    }
