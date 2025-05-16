import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from __future__ import annotations
    import warnings
    warnings.filterwarnings("ignore")
    import json
    from pathlib import Path
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from concurrent.futures import ThreadPoolExecutor
    import itertools
    import marimo

    import plotly.graph_objects as go
    return go, marimo, np, pd, plt


@app.cell
def _(marimo, pd):
    verif_df_1 = pd.read_csv((marimo.notebook_location() / "public" / "verif_df1.csv"), compression=None)
    verif_df_2 = pd.read_csv((marimo.notebook_location() / "public" / "verif_df2.csv"), compression=None)
    verif_df = pd.concat([verif_df_1, verif_df_2], ignore_index=True)
    return (verif_df,)


@app.cell
def _(plt):
    def plot_verification(verif_df_avg, models, variables, metrics):
        for param in variables.keys():

            for metric in metrics:
                fig, ax = plt.subplots()
                for model_name in models:
                    verif_df_avg.loc[model_name].loc[param].plot(
                        y=[metric], marker="o", title=f"{param} - {metric}", label=[model_name], ax=ax, legend=False
                    )
                    ax.set_ylabel(f"{metric} [{variables[param]}]")
                    ax.set_xlabel("lead time [h]")
                plt.legend(prop={'size': 8})
                plt.show()
                plt.clf(); plt.cla(); plt.close()
    return


@app.cell
def _(go, np, pd):
    def plot_skill_heatmap(
        verif_df        : pd.DataFrame,
        *,
        model           : str,                     # model to evaluate
        baseline        : str = "COSMO-E",         # reference model
        metric          : str = "rmse",            # column to compare
    ) -> go.Figure:
        """
        1-row-per-variable × 1-column-per-day heat-map.
          • colour  = % improvement in RMSE vs baseline  (blue = better)
          • hover   = baseline RMSE, model RMSE, % improvement
        """

        # ────────────────────────────────────────────────────────────────────────
        # 1) average over reftimes
        avg = (
            verif_df
            .groupby(["model_name", "param", "lead_time"])
            .mean(numeric_only=True)[metric]
        )

        # 2) pivot to (param, lead_time)
        m_tab   = avg.xs(model,    level="model_name").unstack("lead_time")
        ref_tab = avg.xs(baseline, level="model_name").unstack("lead_time")
        # keep only common lead-times & variables
        common_leads = sorted(set(m_tab.columns) & set(ref_tab.columns))
        m_tab, ref_tab = m_tab[common_leads], ref_tab[common_leads]

        common_vars = m_tab.index.intersection(ref_tab.index)
        m_tab, ref_tab = m_tab.loc[common_vars], ref_tab.loc[common_vars]

        # 3) aggregate 6-hourly → daily means
        day_numbers = (np.array(common_leads) // 24 + 1).astype(int)
        m_tab.columns, ref_tab.columns = day_numbers, day_numbers

        m_day, ref_day = (
            t.groupby(level=0, axis=1).mean() for t in (m_tab, ref_tab)
        )

        # 4) % improvement
        skill_pct = (ref_day - m_day) / ref_day * 100.0

        # 7) colour range symmetric
        vmax = float(np.nanmax(np.abs(skill_pct.values)) or 1e-9)
        baseline_np = ref_day.to_numpy(dtype=float)
        model_np    = m_day  .to_numpy(dtype=float)
        customdata  = np.dstack([baseline_np, model_np])

        # --- figure ---------------------------------------------------------------
        fig = go.Figure(
            go.Heatmap(
                z          = skill_pct.values,
                x          = skill_pct.columns,
                y          = skill_pct.index,
                customdata = customdata,
                colorscale = "RdBu",
                zmin = -vmax,
                zmax = vmax,
                hovertemplate = (
                    "<b>%{y}</b><br>"
                    "Day-%{x}<br>"
                    "Baseline&nbsp;RMSE: %{customdata[0]:.3f}<br>"
                    "Model&nbsp;&nbsp;&nbsp;RMSE: %{customdata[1]:.3f}<br>"
                    "<b>%{z:.1f}%</b> improvement"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title           = f"{metric.upper()} Percentage improvement vs {baseline}",
            xaxis_title     = "Days out",
            yaxis_title     = "",
            yaxis_autorange = "reversed",
            template        = "none",
            xaxis_showgrid  = False,
            yaxis_showgrid  = False,
            xaxis_zeroline  = False,
            yaxis_zeroline  = False,
            margin          = dict(l=40, r=40, t=60, b=40),   # ← extra padding
        )
        return fig


    return (plot_skill_heatmap,)


@app.cell
def _(marimo, verif_df):
    select = marimo.ui.dropdown(
        verif_df.model_name.unique(), 
        value='stage_C-metno_low_lam-rollout', 
        label='Choose Model Config')
    select
    return (select,)


@app.cell
def _(plot_skill_heatmap, select, verif_df):
    fig = plot_skill_heatmap(
        verif_df,
        model=select.value,
        baseline="COSMO-E",
        metric="rmse",
    )
    fig

    return


if __name__ == "__main__":
    app.run()
