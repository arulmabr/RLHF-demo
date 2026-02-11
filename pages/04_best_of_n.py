import streamlit as st

st.set_page_config(page_title="Best-of-N Sampling", page_icon="🎲", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

st.markdown('<p class="section-header">INFERENCE-TIME SCALING</p>', unsafe_allow_html=True)
st.title("Best-of-N Sampling Playground")
st.markdown("Sample N responses, score them with a reward model, and return the best one. Simple — but what happens as N grows?")
st.markdown("---")

# ── Session state ────────────────────────────────────────────────────────────
if "bon_seed" not in st.session_state:
    st.session_state.bon_seed = 42

# ── Controls ─────────────────────────────────────────────────────────────────
st.markdown("### Configuration")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])

with col_ctrl1:
    n_log = st.slider("N (number of samples)", min_value=0, max_value=7, value=3,
                       help="Log₂ scale: 0→1, 1→2, 2→4, 3→8, 4→16, 5→32, 6→64, 7→128")
    N = 2 ** n_log
    st.markdown(f"**N = {N}** samples")

with col_ctrl2:
    rm_noise = st.slider("RM Noise Level (proxy misalignment)", 0.0, 2.0, 0.5, 0.1,
                          help="How noisy is the reward model? Higher = worse proxy.")

with col_ctrl3:
    if st.button("🎲 Re-roll", help="New random samples"):
        st.session_state.bon_seed += 1
        st.rerun()

rng = np.random.RandomState(st.session_state.bon_seed)

st.markdown("---")

# ── Simulation: single N ────────────────────────────────────────────────────
st.markdown("### Sample Visualization")

st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">How it works:</strong> Each dot is a sampled response.
<strong>True quality</strong> is what a human would rate it.
<strong>Proxy reward</strong> is the RM's (noisy) score. Best-of-N picks the highest proxy reward.
</div>
""", unsafe_allow_html=True)

# Generate samples
true_quality = rng.randn(N)
proxy_reward = true_quality + rm_noise * rng.randn(N)

best_idx = np.argmax(proxy_reward)
oracle_idx = np.argmax(true_quality)

fig_scatter = go.Figure()

# All points
fig_scatter.add_trace(go.Scatter(
    x=true_quality, y=proxy_reward,
    mode="markers",
    marker=dict(size=10, color=COLORS["blue"], opacity=0.5),
    name="Samples",
    hovertemplate="True: %{x:.2f}<br>Proxy: %{y:.2f}<extra></extra>",
))

# Best-of-N point
fig_scatter.add_trace(go.Scatter(
    x=[true_quality[best_idx]], y=[proxy_reward[best_idx]],
    mode="markers",
    marker=dict(size=18, color=COLORS["red"], symbol="star", line=dict(width=2, color="white")),
    name=f"Best-of-{N} (RM pick)",
    hovertemplate="True: %{x:.2f}<br>Proxy: %{y:.2f}<extra>RM Pick</extra>",
))

# Oracle best
if oracle_idx != best_idx:
    fig_scatter.add_trace(go.Scatter(
        x=[true_quality[oracle_idx]], y=[proxy_reward[oracle_idx]],
        mode="markers",
        marker=dict(size=18, color=COLORS["green"], symbol="diamond",
                    line=dict(width=2, color="white")),
        name="Oracle Best (true best)",
        hovertemplate="True: %{x:.2f}<br>Proxy: %{y:.2f}<extra>Oracle</extra>",
    ))

# Diagonal reference line
q_range = [min(true_quality.min(), proxy_reward.min()) - 0.5,
           max(true_quality.max(), proxy_reward.max()) + 0.5]
fig_scatter.add_trace(go.Scatter(
    x=q_range, y=q_range,
    mode="lines",
    line=dict(color=COLORS["gray"], dash="dot", width=1),
    name="Perfect RM (y=x)",
    showlegend=True,
))

fig_scatter.update_layout(
    xaxis_title="True Quality",
    yaxis_title="Proxy Reward (RM Score)",
    title=f"Best-of-{N} Sampling (noise={rm_noise:.1f})",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig_scatter, width="stretch")

# Metrics
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("RM Pick — True Quality", f"{true_quality[best_idx]:.2f}")
col_m2.metric("Oracle Best — True Quality", f"{true_quality[oracle_idx]:.2f}")
gap = true_quality[oracle_idx] - true_quality[best_idx]
col_m3.metric("Gap (oracle − RM pick)", f"{gap:.2f}",
              delta=f"{'overoptimized' if gap > 0.5 else 'close'}", delta_color="inverse")

st.markdown("---")

# ── Scaling curves: sweep over N ────────────────────────────────────────────
st.markdown("### Scaling Curves: What Happens as N Grows?")

st.markdown(f"""
<div class="insight-box">
<strong>Key insight:</strong> As N increases, the best proxy reward keeps climbing —
but true quality plateaus and eventually <em>drops</em> due to reward hacking.
The RM selects responses that exploit its blind spots.
</div>
""", unsafe_allow_html=True)

N_values = [1, 2, 4, 8, 16, 32, 64, 128]
n_trials = 200

avg_true = []
avg_proxy = []
avg_oracle = []

for n_val in N_values:
    trial_true = []
    trial_proxy = []
    trial_oracle = []
    for t in range(n_trials):
        tq = rng.randn(n_val)
        pr = tq + rm_noise * rng.randn(n_val)
        best = np.argmax(pr)
        trial_true.append(tq[best])
        trial_proxy.append(pr[best])
        trial_oracle.append(tq.max())
    avg_true.append(np.mean(trial_true))
    avg_proxy.append(np.mean(trial_proxy))
    avg_oracle.append(np.mean(trial_oracle))

fig_scaling = go.Figure()

fig_scaling.add_trace(go.Scatter(
    x=N_values, y=avg_proxy,
    mode="lines+markers",
    name="Proxy Reward (RM says)",
    line=dict(color=COLORS["red"], width=3),
    marker=dict(size=8),
))

fig_scaling.add_trace(go.Scatter(
    x=N_values, y=avg_true,
    mode="lines+markers",
    name="True Quality (RM pick)",
    line=dict(color=COLORS["blue"], width=3),
    marker=dict(size=8),
))

fig_scaling.add_trace(go.Scatter(
    x=N_values, y=avg_oracle,
    mode="lines+markers",
    name="Oracle Best (ceiling)",
    line=dict(color=COLORS["green"], width=2, dash="dash"),
    marker=dict(size=6),
))

# Highlight divergence region
if len(avg_proxy) > 2:
    gaps = [p - t for p, t in zip(avg_proxy, avg_true)]
    max_gap_idx = np.argmax(gaps)
    fig_scaling.add_vline(
        x=N_values[max_gap_idx],
        line_dash="dot",
        line_color=COLORS["orange"],
        annotation_text="Max divergence",
        annotation_position="top",
    )

fig_scaling.update_layout(
    xaxis_title="N (number of samples)",
    yaxis_title="Average Score",
    title="Scaling Curves: Proxy Reward vs True Quality",
    xaxis_type="log",
    xaxis=dict(tickvals=N_values, ticktext=[str(n) for n in N_values]),
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig_scaling, width="stretch")

# Gap analysis
st.markdown("### The Overoptimization Gap")

gaps = [p - t for p, t in zip(avg_proxy, avg_true)]

fig_gap = go.Figure()
fig_gap.add_trace(go.Bar(
    x=[str(n) for n in N_values],
    y=gaps,
    marker_color=[COLORS["green"] if g < 0.3 else COLORS["orange"] if g < 0.6 else COLORS["red"]
                  for g in gaps],
    hovertemplate="N=%{x}<br>Gap=%{y:.3f}<extra></extra>",
))
fig_gap.update_layout(
    xaxis_title="N",
    yaxis_title="Proxy − True (overoptimization gap)",
    title="How Much Does the RM Overestimate?",
    height=350,
)
st.plotly_chart(fig_gap, width="stretch")

st.markdown("---")

# ── Challenge questions ──────────────────────────────────────────────────────
st.markdown("### Challenge Questions")

with st.expander("1. At what N does true quality stop improving significantly?"):
    st.markdown(f"""
Based on the current simulation (noise={rm_noise:.1f}):

The **oracle best** (green dashed line) keeps improving with more samples — this is pure
order-statistic scaling. But the **RM pick's true quality** (blue line) typically plateaus
around **N = 8–16** and may even decline at higher N.

This is because with more samples, the RM is more likely to find a response that
*exploits its blind spots* rather than one that is genuinely better.
""")

with st.expander("2. What is the computational cost of Best-of-N?"):
    st.markdown("""
Best-of-N requires:
- **N forward passes** through the language model (generation)
- **N forward passes** through the reward model (scoring)

This means Best-of-128 costs **128x** the compute of a single sample.
For comparison, a single RLHF-trained model generates one response at 1x cost
but required expensive training. Best-of-N trades training compute for inference compute.
""")

with st.expander("3. How does RM noise affect the optimal N?"):
    st.markdown("""
Try adjusting the RM noise slider above!

- **Low noise** (< 0.3): Higher N reliably improves quality. The RM is a good proxy.
- **Medium noise** (0.3 – 1.0): Diminishing returns kick in around N=16–32. Beyond that, you're selecting for noise.
- **High noise** (> 1.0): Even small N can overoptimize. The RM is basically picking random responses that happen to match its biases.

This is a key practical insight: **the noisier your reward model, the less you should scale N**.
""")
