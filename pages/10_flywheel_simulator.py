import streamlit as st

st.set_page_config(page_title="The Self-Improvement Flywheel", page_icon="🔄", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">DATA FLYWHEELS &amp; SELF-IMPROVING SYSTEMS</p>', unsafe_allow_html=True)
st.title("The Self-Improvement Flywheel")
st.markdown("Compare how different training signals — human judgment, AI judges, and formal verification — drive self-improvement loops over time.")
st.markdown("---")

# ── Signal type definitions ─────────────────────────────────────────────────
SIGNAL_TYPES = {
    "Human Judgment (RLHF)": {
        "ceiling": 0.82,
        "learning_rate": 0.08,
        "rm_decay": 0.012,
        "color": COLORS["blue"],
        "desc": "Capped by human evaluator quality; reward model drifts as policy shifts.",
    },
    "AI Judge (RLAIF)": {
        "ceiling": 0.75,
        "learning_rate": 0.12,
        "rm_decay": 0.018,
        "color": COLORS["purple"],
        "desc": "Faster iteration, but lower ceiling; judge degrades faster than human.",
    },
    "Verification (RLVR)": {
        "ceiling": 0.98,
        "learning_rate": 0.05,
        "rm_decay": 0.0,
        "color": COLORS["green"],
        "desc": "Deterministic signal — no decay. Slow per step but keeps climbing.",
    },
}

# ── Controls ────────────────────────────────────────────────────────────────
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    signal_type = st.selectbox("Signal Type", list(SIGNAL_TYPES.keys()))
with col_c2:
    num_rounds = st.slider("Training Rounds", min_value=1, max_value=50, value=30)
with col_c3:
    noise_level = st.slider("Signal Noise", min_value=0.0, max_value=0.15, value=0.03, step=0.01)
with col_c4:
    show_all = st.checkbox("Show all signal types", value=False)

# ── Simulation ──────────────────────────────────────────────────────────────
def simulate_flywheel(cfg, rounds, noise, seed=42):
    rng = np.random.RandomState(seed)
    quality = [0.30]  # starting model quality
    reliability = [1.0]  # starting signal reliability

    for r in range(rounds):
        current_q = quality[-1]
        current_rel = reliability[-1]
        gap = cfg["ceiling"] - current_q
        improvement = cfg["learning_rate"] * gap * current_rel + rng.randn() * noise
        new_q = np.clip(current_q + improvement, 0.0, 1.0)
        quality.append(new_q)

        # Reward model decay: reliability drops as distribution shifts
        new_rel = max(0.1, current_rel - cfg["rm_decay"])
        reliability.append(new_rel)

    return np.array(quality), np.array(reliability)


types_to_plot = list(SIGNAL_TYPES.keys()) if show_all else [signal_type]

results = {}
for st_name in types_to_plot:
    cfg = SIGNAL_TYPES[st_name]
    q, rel = simulate_flywheel(cfg, num_rounds, noise_level)
    results[st_name] = {"quality": q, "reliability": rel, "cfg": cfg}

# ── Chart 1: Model Quality Over Rounds ──────────────────────────────────────
fig_quality = go.Figure()
rounds_x = np.arange(num_rounds + 1)

for st_name, data in results.items():
    cfg = data["cfg"]
    fig_quality.add_trace(go.Scatter(
        x=rounds_x, y=data["quality"],
        mode="lines", name=st_name,
        line=dict(color=cfg["color"], width=3),
    ))
    # ceiling line
    fig_quality.add_hline(
        y=cfg["ceiling"], line_dash="dot", line_color=cfg["color"],
        opacity=0.4,
    )

fig_quality.update_layout(
    title="Model Quality Over Training Rounds",
    xaxis_title="Round",
    yaxis_title="Model Quality",
    yaxis=dict(range=[0, 1.05]),
    height=420,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    margin=dict(b=80),
)
st.plotly_chart(fig_quality, use_container_width=True)

# ── Chart 2: Signal Reliability Over Rounds ─────────────────────────────────
fig_rel = go.Figure()

for st_name, data in results.items():
    cfg = data["cfg"]
    fig_rel.add_trace(go.Scatter(
        x=rounds_x, y=data["reliability"],
        mode="lines", name=st_name,
        line=dict(color=cfg["color"], width=3),
    ))

fig_rel.update_layout(
    title="Signal Reliability Over Rounds (Reward Model Decay)",
    xaxis_title="Round",
    yaxis_title="Signal Reliability",
    yaxis=dict(range=[0, 1.1]),
    height=350,
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    margin=dict(b=80),
)
st.plotly_chart(fig_rel, use_container_width=True)

# ── Metrics row ─────────────────────────────────────────────────────────────
st.markdown("#### Performance Summary")
metric_cols = st.columns(len(types_to_plot))

for col, st_name in zip(metric_cols, types_to_plot):
    data = results[st_name]
    cfg = data["cfg"]
    final_q = data["quality"][-1]
    target_90 = cfg["ceiling"] * 0.9
    rounds_to_90 = None
    for i, q in enumerate(data["quality"]):
        if q >= target_90:
            rounds_to_90 = i
            break
    # Signal efficiency: final quality per unit of decay
    total_decay = 1.0 - data["reliability"][-1]
    efficiency = final_q / max(total_decay, 0.01)

    with col:
        st.markdown(f"**{st_name}**")
        st.metric("Final Quality", f"{final_q:.3f}")
        st.metric("Rounds to 90% Ceiling", f"{rounds_to_90 if rounds_to_90 is not None else '>' + str(num_rounds)}")
        st.metric("Signal Efficiency", f"{efficiency:.1f}")

# ── Description of selected signal ──────────────────────────────────────────
st.markdown("---")
st.markdown("#### Signal Type Details")
for st_name in types_to_plot:
    cfg = SIGNAL_TYPES[st_name]
    st.markdown(f"""
<div class="concept-card">
<strong style="color:{cfg['color']};">{st_name}</strong><br/>
<span style="color:{COLORS['gray']};">Ceiling: {cfg['ceiling']:.2f} &bull; Learning Rate: {cfg['learning_rate']} &bull; RM Decay: {cfg['rm_decay']}</span><br/>
{cfg['desc']}
</div>
""", unsafe_allow_html=True)

# ── Insight ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="insight-box">
<strong>Key insight:</strong> Judgment-based loops (RLHF, RLAIF) are fundamentally capped by the quality of
the evaluator. As the policy improves and moves off-distribution, the reward model degrades &mdash;
creating a ceiling. Verification-based loops (RLVR) use deterministic signals that never decay,
so the model can keep improving as long as the task is verifiable. The tradeoff: verification only
works for domains with checkable answers (math, code, formal proofs), while judgment scales to
open-ended tasks like creative writing or summarization.
</div>
""", unsafe_allow_html=True)
