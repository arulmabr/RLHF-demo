import streamlit as st
import numpy as np
import plotly.graph_objects as go

from style import inject_custom_css, COLORS, softmax, kl_divergence, entropy

# ── Page config & styling ────────────────────────────────────────────────────
inject_custom_css()

st.markdown('<p class="section-header">SECTION III</p>', unsafe_allow_html=True)
st.title("KL Divergence")
st.markdown("**The most important quantity in post-training.**")

# ── 1. Explanation ────────────────────────────────────────────────────────────
st.markdown("---")
st.header("What Is KL Divergence?")

st.markdown("""
KL divergence (Kullback-Leibler divergence) measures how much **extra information**
is needed when you use distribution $Q$ to encode data that actually comes from
distribution $P$.  Think of it as the "cost of being wrong" about the true
distribution.
""")

st.markdown("""
<div class="big-formula">
KL(P || Q) &nbsp;=&nbsp; &Sigma;<sub>x</sub> &nbsp; P(x) &middot; log &nbsp; P(x) / Q(x)
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
<div class="concept-card">
<strong>Three key properties</strong><br/>
1. <strong>Non-negative:</strong> KL(P||Q) &ge; 0, with equality iff P = Q.<br/>
2. <strong>Not symmetric:</strong> KL(P||Q) &ne; KL(Q||P) in general.<br/>
3. <strong>Units:</strong> nats (base e) or bits (base 2).
</div>
""", unsafe_allow_html=True)
with col_b:
    st.markdown("""
<div class="concept-card">
<strong>Intuition</strong><br/>
Imagine you designed a Morse code optimised for Q.  If the true source is P,
you waste exactly KL(P||Q) extra bits per symbol on average.  It counts the
<em>surprise gap</em> between what you expected and what actually happened.
</div>
""", unsafe_allow_html=True)


# ── 2. Interactive KL Calculator ─────────────────────────────────────────────
st.markdown("---")
st.header("Interactive KL Calculator")
st.markdown("Define two discrete distributions over 5 categories.  Sliders are "
            "auto-normalised so they always sum to 1.")

categories = ["A", "B", "C", "D", "E"]
n_cats = len(categories)

col_p, col_q = st.columns(2)

raw_p = []
with col_p:
    st.markdown("##### Distribution P")
    for i, cat in enumerate(categories):
        raw_p.append(st.slider(f"P({cat})", 0.01, 10.0, 2.0,
                               step=0.01, key=f"p_{i}"))

raw_q = []
with col_q:
    st.markdown("##### Distribution Q")
    for i, cat in enumerate(categories):
        raw_q.append(st.slider(f"Q({cat})", 0.01, 10.0, 2.0,
                               step=0.01, key=f"q_{i}"))

p = np.array(raw_p, dtype=np.float64)
p = p / p.sum()
q = np.array(raw_q, dtype=np.float64)
q = q / q.sum()

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

# -- Metrics row
m1, m2, m3 = st.columns(3)
m1.metric("KL(P || Q)", f"{kl_pq:.4f} nats")
m2.metric("KL(Q || P)", f"{kl_qp:.4f} nats")
m3.metric("Asymmetry gap", f"{abs(kl_pq - kl_qp):.4f} nats")

# -- Overlaid bar chart
bar_x = np.arange(n_cats)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=categories, y=p, name="P",
    marker_color=COLORS["blue"], opacity=0.8,
))
fig_bar.add_trace(go.Bar(
    x=categories, y=q, name="Q",
    marker_color=COLORS["red"], opacity=0.8,
))
fig_bar.update_layout(
    barmode="group",
    title="P vs Q (normalised)",
    xaxis_title="Category",
    yaxis_title="Probability",
    height=370,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_bar, use_container_width=True)

# -- Per-token contribution breakdown
contributions_pq = p * np.log(p / np.clip(q, 1e-10, None))
contributions_qp = q * np.log(q / np.clip(p, 1e-10, None))

fig_contrib = go.Figure()
fig_contrib.add_trace(go.Bar(
    x=categories, y=contributions_pq, name="KL(P||Q) contribution",
    marker_color=COLORS["blue"],
))
fig_contrib.add_trace(go.Bar(
    x=categories, y=contributions_qp, name="KL(Q||P) contribution",
    marker_color=COLORS["red"],
))
fig_contrib.update_layout(
    barmode="group",
    title="Per-Category KL Contributions",
    xaxis_title="Category",
    yaxis_title="Contribution (nats)",
    height=340,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_contrib, use_container_width=True)

st.markdown("""
<div class="insight-box">
<strong>Notice the asymmetry!</strong>&ensp; KL(P||Q) penalises places where P has
mass but Q does not.  KL(Q||P) does the opposite.  This asymmetry is
<em>exactly</em> why forward KL and reverse KL lead to different model behaviour in
post-training.
</div>
""", unsafe_allow_html=True)


# ── 3. Forward vs Reverse KL Visualiser ──────────────────────────────────────
st.markdown("---")
st.header("Forward vs Reverse KL Visualiser")
st.markdown("""
The **target** P is a bimodal mixture of two Gaussians. The **approximation** Q is
a single Gaussian whose mean and standard deviation you control.  Watch how the
optimal Q differs depending on which KL direction you minimise.
""")

# Target: mixture of two Gaussians
MU1, MU2, SIG1, SIG2 = -2.5, 2.5, 0.9, 0.9
x_range = np.linspace(-7, 7, 500)


def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def p_target(x):
    return 0.5 * gaussian(x, MU1, SIG1) + 0.5 * gaussian(x, MU2, SIG2)


p_vals = p_target(x_range)

kl_direction = st.radio(
    "KL direction to minimise",
    ["Forward KL  :  KL(P || Q)  -- mean-seeking (SFT behaviour)",
     "Reverse KL  :  KL(Q || P)  -- mode-seeking (RLHF behaviour)"],
    index=0,
)
is_forward = kl_direction.startswith("Forward")

col_mu, col_sig = st.columns(2)
with col_mu:
    q_mu = st.slider("Q mean", -6.0, 6.0,
                      0.0 if is_forward else MU1,
                      step=0.1, key="q_mu_fwdrev")
with col_sig:
    q_sig = st.slider("Q std dev", 0.3, 5.0,
                       2.5 if is_forward else 0.9,
                       step=0.1, key="q_sig_fwdrev")

q_vals = gaussian(x_range, q_mu, q_sig)

# Numerical KL on the grid (approximate)
dx = x_range[1] - x_range[0]
p_safe = np.clip(p_vals, 1e-10, None)
q_safe = np.clip(q_vals, 1e-10, None)
fwd_kl = np.sum(p_safe * np.log(p_safe / q_safe)) * dx
rev_kl = np.sum(q_safe * np.log(q_safe / p_safe)) * dx

fig_fwdrev = go.Figure()
fig_fwdrev.add_trace(go.Scatter(
    x=x_range, y=p_vals, mode="lines", name="P (target, bimodal)",
    line=dict(color=COLORS["blue"], width=3),
))
fig_fwdrev.add_trace(go.Scatter(
    x=x_range, y=q_vals, mode="lines", name="Q (approximation)",
    line=dict(color=COLORS["red"], width=3, dash="dash"),
))

# Shade the gap
fig_fwdrev.add_trace(go.Scatter(
    x=np.concatenate([x_range, x_range[::-1]]),
    y=np.concatenate([np.minimum(p_vals, q_vals),
                      np.maximum(p_vals, q_vals)[::-1]]),
    fill="toself",
    fillcolor="rgba(243, 156, 18, 0.12)",
    line=dict(width=0),
    name="Mismatch region",
    hoverinfo="skip",
))

active_kl = fwd_kl if is_forward else rev_kl
direction_label = "KL(P||Q)" if is_forward else "KL(Q||P)"
fig_fwdrev.update_layout(
    title=f"{direction_label} = {active_kl:.3f} nats",
    xaxis_title="x",
    yaxis_title="Density",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_fwdrev, use_container_width=True)

col_fwd_info, col_rev_info = st.columns(2)
with col_fwd_info:
    st.markdown(f"""
<div class="concept-card">
<strong>Forward KL &nbsp; KL(P||Q) = {fwd_kl:.3f}</strong><br/>
Penalises Q wherever <strong>P &gt; 0</strong>.  Q must cover <em>all</em> modes of P,
even if it wastes probability mass in between.<br/>
<span style="color:{COLORS['green']};">&#x2192; Mean-seeking &rarr; SFT behaviour</span>
</div>
""", unsafe_allow_html=True)
with col_rev_info:
    st.markdown(f"""
<div class="concept-card">
<strong>Reverse KL &nbsp; KL(Q||P) = {rev_kl:.3f}</strong><br/>
Penalises Q wherever <strong>Q &gt; 0 but P &approx; 0</strong>.  Q collapses onto a
single mode to avoid placing mass where P is zero.<br/>
<span style="color:{COLORS['orange']};">&#x2192; Mode-seeking &rarr; RLHF behaviour</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
<strong>Try it!</strong>&ensp; For Forward KL, set Q mean &asymp; 0 and Q std &asymp; 3
to cover both modes.  For Reverse KL, set Q mean &asymp; &minus;2.5 (or +2.5) and Q
std &asymp; 0.9 to lock onto one mode.  Notice how the optimal strategy
<em>flips</em>.
</div>
""", unsafe_allow_html=True)


# ── 4. KL Budget Explorer ────────────────────────────────────────────────────
st.markdown("---")
st.header("KL Budget Explorer")
st.markdown("""
Every post-training method implicitly or explicitly **spends a KL budget** away
from the base model.  Below, see where KL appears in each technique and explore
the quality-vs-KL tradeoff (Goodhart's Law in action).
""")

# -- Methods table
st.markdown("##### Where KL Appears in Each Method")
st.markdown("""
| Method | How KL enters | Typical control knob |
|--------|--------------|---------------------|
| **SFT** | Implicit: cross-entropy loss pushes model toward data distribution. KL from base grows with training steps. | Learning rate, epochs |
| **Distillation** | Forward KL between teacher and student soft outputs. | Temperature $\\tau$ |
| **RLHF / PPO** | Explicit KL penalty: $R_{\\text{total}} = R(y) - \\beta\\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ | $\\beta$ coefficient |
| **DPO** | Implicit KL: $\\beta$ in the DPO objective controls divergence from reference. | $\\beta$ parameter |
| **Best-of-N** | Sampling N candidates incurs KL $\\approx \\log N - (N-1)/N$.  | N (sample count) |
""")

st.markdown("##### Quality vs KL: The Goodhart Curve")
st.markdown("Use the slider to set a hypothetical KL budget and see how **true "
            "quality** and **proxy (learned) reward** evolve.")

kl_budget = st.slider("KL budget (nats)", 0.0, 30.0, 10.0, step=0.5,
                       key="kl_budget")
beta_ctrl = st.slider("Beta (KL penalty weight)", 0.01, 1.0, 0.1, step=0.01,
                       key="beta_ctrl")

kl_axis = np.linspace(0, 30, 300)

# Proxy reward: keeps climbing (overfits)
proxy_reward = 4.0 * (1 - np.exp(-0.15 * kl_axis)) + 0.08 * kl_axis

# True quality: peaks then drops (Goodhart)
true_quality = 5.0 * (1 - np.exp(-0.2 * kl_axis)) - 0.012 * kl_axis ** 1.5
true_quality = np.maximum(true_quality, 0)

# KL-penalised objective
penalised = proxy_reward - beta_ctrl * kl_axis

fig_goodhart = go.Figure()
fig_goodhart.add_trace(go.Scatter(
    x=kl_axis, y=true_quality, mode="lines",
    name="True quality",
    line=dict(color=COLORS["green"], width=3),
))
fig_goodhart.add_trace(go.Scatter(
    x=kl_axis, y=proxy_reward, mode="lines",
    name="Proxy reward (learned RM)",
    line=dict(color=COLORS["red"], width=2, dash="dash"),
))
fig_goodhart.add_trace(go.Scatter(
    x=kl_axis, y=penalised, mode="lines",
    name=f"Penalised objective (beta={beta_ctrl:.2f})",
    line=dict(color=COLORS["cyan"], width=2, dash="dot"),
))

# Vertical line at KL budget
fig_goodhart.add_vline(
    x=kl_budget, line_width=2, line_dash="dash",
    line_color=COLORS["orange"],
    annotation_text=f"KL budget = {kl_budget:.1f}",
    annotation_position="top right",
    annotation_font_color=COLORS["orange"],
)

# Mark optimal true quality
peak_idx = int(np.argmax(true_quality))
fig_goodhart.add_trace(go.Scatter(
    x=[kl_axis[peak_idx]], y=[true_quality[peak_idx]],
    mode="markers+text",
    marker=dict(size=12, color=COLORS["green"], symbol="star"),
    text=["Peak true quality"],
    textposition="top center",
    textfont=dict(color=COLORS["green"], size=11),
    showlegend=False,
))

fig_goodhart.update_layout(
    title="Goodhart's Law: Quality vs KL Divergence",
    xaxis_title="KL from base model (nats)",
    yaxis_title="Score",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_goodhart, use_container_width=True)

# -- Values at the chosen budget
budget_idx = np.argmin(np.abs(kl_axis - kl_budget))
v1, v2, v3 = st.columns(3)
v1.metric("True quality @ budget",
          f"{true_quality[budget_idx]:.2f}")
v2.metric("Proxy reward @ budget",
          f"{proxy_reward[budget_idx]:.2f}")
v3.metric("Penalised objective @ budget",
          f"{penalised[budget_idx]:.2f}")

st.markdown(f"""
<div class="concept-card">
<strong>Reading the chart</strong><br/>
&bull; <span style="color:{COLORS['green']};">True quality</span> peaks around
KL &asymp; {kl_axis[peak_idx]:.0f} nats then <em>drops</em> &mdash; this is
<strong>Goodhart's Law</strong>: optimising too hard against a proxy overshoots
the true objective.<br/>
&bull; <span style="color:{COLORS['red']};">Proxy reward</span> keeps climbing
because the learned reward model doesn't know it's being exploited.<br/>
&bull; <span style="color:{COLORS['cyan']};">The penalised objective</span>
(proxy &minus; &beta;&middot;KL) creates a natural stopping point.  Larger &beta;
pulls the peak <em>left</em> (more conservative).  Try moving &beta; to see this.
</div>
""", unsafe_allow_html=True)


# ── 5. Key Insight ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="insight-box" style="font-size:1.1rem; padding:24px 28px;">
<strong>Key Insight</strong><br/><br/>
KL divergence is the <strong>currency of post-training</strong>.&ensp; Every
method &mdash; SFT, distillation, RLHF, DPO, Best-of-N &mdash; spends a KL
budget to move the model away from its base distribution toward higher-quality
behaviour.<br/><br/>
<strong>&beta; controls the exchange rate.</strong>&ensp; A small &beta; lets the
model spend KL freely (high reward, risk of Goodhart).  A large &beta; is
conservative (stays close to base, less reward-hacking).<br/><br/>
The art of post-training is choosing <em>how much</em> KL to spend and
<em>where</em> to spend it.
</div>
""", unsafe_allow_html=True)
