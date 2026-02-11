import torch
import torch.distributions as dis
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
# setup distributions & true value
p = dis.Normal(loc=0., scale=1.)
q = dis.Normal(loc=0.1, scale=1.)

true_kl = dis.kl_divergence(p, q).item()
print(f"true KL divergence: {true_kl:.6f}")

# logarithmic sample sizes → smoother convergence curves
sample_sizes = np.logspace(1, 8, num=30, dtype=int)

# result collectors
rel_bias = {'k1': [], 'k2': [], 'k3': []}
rel_std  = {'k1': [], 'k2': [], 'k3': []}

for n in sample_sizes:
    x = q.sample((n,))
    logr = p.log_prob(x) - q.log_prob(x)

    k1 = -logr
    k2 = logr ** 2 / 2
    k3 = logr.expm1() - logr   # expm1 = exp(x) - 1 more stable

    for name, k in zip(['k1','k2','k3'], [k1, k2, k3]):
        mean_est = k.mean().item()
        std_est  = k.std().item()
        rel_bias[name].append((mean_est - true_kl) / true_kl)
        rel_std[name].append(std_est / true_kl)

    print(f"finished n = {n:>10,d}")

# ──────────────────────────────────────────────────────────────
# now the fun part — interactive plotly figure with dark theme

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Absolute Relative Bias of KL Estimates",
        "Relative Standard Deviation of Estimates"
    ),
    horizontal_spacing=0.08
)

# names & nice colors (work great on dark bg)
names = ['-log r  (k1)', '(log r)² / 2  (k2)', 'exp(log r)-1 - log r  (k3)']
colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']   # coral, teal, soft yellow

# bias (absolute relative error)
for name, color, lbl in zip(['k1','k2','k3'], colors, names):
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=np.abs(rel_bias[name]),
            mode='lines+markers',
            name=lbl,
            line=dict(color=color, width=2.8),
            marker=dict(size=7, opacity=0.7),
            hovertemplate='n = %{x:,}<br>|bias| = %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

# std (relative uncertainty)
for name, color, lbl in zip(['k1','k2','k3'], colors, names):
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=rel_std[name],
            mode='lines+markers',
            name=lbl,
            line=dict(color=color, width=2.8),
            marker=dict(size=7, opacity=0.7),
            hovertemplate='n = %{x:,}<br>rel std = %{y:.4f}<extra></extra>',
            showlegend=False   # only show legend once
        ),
        row=1, col=2
    )

# layout — dark mode + modern look
fig.update_layout(
    template="plotly_dark",
    height=1000,
    width=2000,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(size=13),
        itemwidth=60
    ),
    hovermode="x unified",
    font=dict(size=14)
)

# axis styling
for col, title in enumerate(["Number of Samples", "Number of Samples"], 1):
    fig.update_xaxes(
        title_text=title,
        type="log",
        row=1, col=col,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        tickfont=dict(size=12)
    )

fig.update_yaxes(
    title_text="| (estimate - true) / true |",
    type="log",
    row=1, col=1,
    gridcolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=12)
)

fig.update_yaxes(
    title_text="std(estimate) / true KL",
    type="log",
    row=1, col=2,
    gridcolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=12)
)

fig.show()