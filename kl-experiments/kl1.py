import torch
import torch.distributions as dis
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# keeping p as bimodal mixture: 0.5 * N(-2,1) + 0.5 * N(2,1)
mix = dis.Categorical(torch.ones(2)/2)
comp = dis.Independent(dis.Normal(torch.tensor([-2.0, 2.0]), torch.ones(2)), 0)
p = dis.MixtureSameFamily(mix, comp)

# initial q params
init_mu = 4.5
init_log_sigma = -1.0  # sigma=1

# opt settings
lr = 0.05
batch_size = 20_000
num_iters = 1000

# iterations to snapshot for evolution plots
snapshot_iters = [0, 10, 50, 100, 200, 500, 1000]

# ──────────────────────────────────────────────────────────────
# reverse kl opt: min KL(q || p) - mode-seeking

mu_r = torch.tensor(init_mu, requires_grad=True)
log_sigma_r = torch.tensor(init_log_sigma, requires_grad=True)
optimizer_r = torch.optim.Adam([mu_r, log_sigma_r], lr=lr)

losses_reverse = []
snapshots_reverse = []  # list of (mu, sigma) at snapshots

for i in range(num_iters + 1):  # +1 to include iter=0
    if i in snapshot_iters:
        snapshots_reverse.append((mu_r.item(), log_sigma_r.exp().item()))

    if i == num_iters:
        break  # don't optimize after last snapshot

    optimizer_r.zero_grad()
    q = dis.Normal(mu_r, log_sigma_r.exp())
    x = q.rsample((batch_size,))
    logq = q.log_prob(x)
    logp = p.log_prob(x)
    loss = (logq - logp).mean()
    loss.backward()
    optimizer_r.step()
    losses_reverse.append(loss.item())

    if i % 100 == 0:
        print(f"reverse iter {i}: loss={loss.item():.4f}, mu={mu_r.item():.2f}, sigma={log_sigma_r.exp().item():.2f}")

print(f"reverse final: mu={mu_r.item():.2f}, sigma={log_sigma_r.exp().item():.2f}")

# ──────────────────────────────────────────────────────────────
# forward kl opt: min KL(p || q) - mean-seeking

mu_f = torch.tensor(init_mu, requires_grad=True)
log_sigma_f = torch.tensor(init_log_sigma, requires_grad=True)
optimizer_f = torch.optim.Adam([mu_f, log_sigma_f], lr=lr)

losses_forward = []
snapshots_forward = []

for i in range(num_iters + 1):
    if i in snapshot_iters:
        snapshots_forward.append((mu_f.item(), log_sigma_f.exp().item()))

    if i == num_iters:
        break

    optimizer_f.zero_grad()
    q = dis.Normal(mu_f, log_sigma_f.exp())
    x = p.sample((batch_size,))
    logp = p.log_prob(x)
    logq = q.log_prob(x)
    loss = (logp - logq).mean()
    loss.backward()
    optimizer_f.step()
    losses_forward.append(loss.item())

    if i % 100 == 0:
        print(f"forward iter {i}: loss={loss.item():.4f}, mu={mu_f.item():.2f}, sigma={log_sigma_f.exp().item():.2f}")

print(f"forward final: mu={mu_f.item():.2f}, sigma={log_sigma_f.exp().item():.2f}")

# ──────────────────────────────────────────────────────────────
# now plot evolution: density traces at snapshots

# x range for pdfs
x_vals = torch.linspace(-5, 5, 500)
p_pdf = p.log_prob(x_vals).exp()  # fixed p density

# create subplots: 2 rows (reverse top, forward bottom), cols = len(snapshots)
n_snaps = len(snapshot_iters)
fig = make_subplots(rows=2, cols=n_snaps,
                    subplot_titles=[f"Iter {it}" for it in snapshot_iters] * 2,
                    vertical_spacing=0.15, horizontal_spacing=0.05)

# colors
color_p = '#FFE566'  # yellow for p
color_q_rev = '#66FFCC'  # teal for reverse q
color_q_fwd = '#FF6692'  # pink for forward q

# add traces for reverse (row 1)
for col, (it, (mu, sigma)) in enumerate(zip(snapshot_iters, snapshots_reverse), 1):
    q = dis.Normal(mu, sigma)
    q_pdf = q.log_prob(x_vals).exp()

    # p density
    fig.add_trace(
        go.Scatter(x=x_vals, y=p_pdf, mode='lines', name='p (fixed)', line=dict(color=color_p, width=2), showlegend=(col==1)),
        row=1, col=col
    )
    # q density
    fig.add_trace(
        go.Scatter(x=x_vals, y=q_pdf, mode='lines', name='q (reverse)', line=dict(color=color_q_rev, width=2), showlegend=(col==1)),
        row=1, col=col
    )

# add traces for forward (row 2)
for col, (it, (mu, sigma)) in enumerate(zip(snapshot_iters, snapshots_forward), 1):
    q = dis.Normal(mu, sigma)
    q_pdf = q.log_prob(x_vals).exp()

    fig.add_trace(
        go.Scatter(x=x_vals, y=p_pdf, mode='lines', name='p (fixed)', line=dict(color=color_p, width=2), showlegend=False),
        row=2, col=col
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=q_pdf, mode='lines', name='q (forward)', line=dict(color=color_q_fwd, width=2), showlegend=(col==1)),
        row=2, col=col
    )

# update axes
for row in [1,2]:
    for col in range(1, n_snaps+1):
        fig.update_xaxes(range=[-5,5], row=row, col=col, showticklabels=(row==2))
        fig.update_yaxes(range=[0, 0.3], row=row, col=col, showticklabels=(col==1))

# row titles
fig.update_annotations(font_size=12)
fig.add_annotation(text="Reverse KL (Mode-Seeking)", xref="paper", yref="paper", x=-0.04, y=0.75, showarrow=False, textangle=-90, font_size=14)
fig.add_annotation(text="Forward KL (Mean-Seeking)", xref="paper", yref="paper", x=-0.04, y=0.25, showarrow=False, textangle=-90, font_size=14)

# overall
fig.update_layout(
    template='plotly_dark',
    title_text="Evolution of q Approximating p During Optimization",
    title_font_size=20,
    title_x=0.5,
    height=1000, width=2000,
    showlegend=True,
    legend=dict(yanchor='top', y=1.05, xanchor='center', x=0.5, orientation='h', bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=80, r=50, t=80, b=50)
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', title_text="Density" if col==1 else "")

fig.show()

# also keep the loss convergence plot from before
loss_fig = go.Figure()

loss_fig.add_trace(go.Scatter(x=list(range(num_iters)), y=losses_reverse, mode='lines', name='Reverse KL (q || p)', line=dict(color=color_q_rev, width=2)))
loss_fig.add_trace(go.Scatter(x=list(range(num_iters)), y=losses_forward, mode='lines', name='Forward KL (p || q)', line=dict(color=color_q_fwd, width=2)))

loss_fig.update_yaxes(type='log', title_text="Estimated KL Loss")
loss_fig.update_xaxes(title_text="Iteration")

loss_fig.update_layout(
    template='plotly_dark',
    title_text="Convergence of KL Losses",
    title_font_size=20,
    title_x=0.5,
    height=500, width=800,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
    font=dict(size=14)
)

loss_fig.show()