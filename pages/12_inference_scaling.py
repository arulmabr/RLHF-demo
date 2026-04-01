import streamlit as st

st.set_page_config(page_title="Inference Scaling Lab", page_icon="🧮", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">DATA FLYWHEELS &amp; SELF-IMPROVING SYSTEMS</p>', unsafe_allow_html=True)
st.title("Inference Scaling Lab")
st.markdown("Explore how generating many samples at inference time — combined with a verifier — can dramatically boost coverage, even with a weak model.")
st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Coverage Calculator (pass@k)", "Compute Tradeoff"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Coverage Calculator
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### pass@k: Coverage from Repeated Sampling")
    st.markdown("If a model has per-sample success rate *p*, what's the chance **at least one** of *k* samples is correct?")

    col_k, col_p, col_v = st.columns(3)
    with col_k:
        is_k = st.slider("Samples (k)", 1, 256, 32, key="is_k")
    with col_p:
        is_p = st.slider("Per-sample success rate (p)", 0.01, 0.50, 0.10, step=0.01, key="is_p")
    with col_v:
        verifier_type = st.selectbox("Verifier Type", ["Oracle (perfect)", "Noisy (90% accurate)", "Weak (70% accurate)"], key="is_verifier")

    # Verifier accuracy
    verifier_acc = {"Oracle (perfect)": 1.0, "Noisy (90% accurate)": 0.90, "Weak (70% accurate)": 0.70}[verifier_type]

    # ── Formula display ─────────────────────────────────────────────────────
    st.markdown(f"""
<div class="big-formula">
pass@k = 1 &minus; (1 &minus; p)<sup>k</sup>
{"&nbsp;&nbsp;&nbsp;(oracle verifier)" if verifier_acc == 1.0 else f"&nbsp;&nbsp;&times; verifier_accuracy<sup>effective</sup> (verifier = {verifier_acc:.0%})"}
</div>
""", unsafe_allow_html=True)

    # ── Compute coverage curves ─────────────────────────────────────────────
    k_range = np.arange(1, 257)

    # Oracle coverage
    oracle_coverage = 1.0 - (1.0 - is_p) ** k_range

    # Adjusted coverage for imperfect verifiers:
    # P(select correct) = P(at least one correct) * P(verifier picks correct | correct exists)
    # With noisy verifier: it might pick wrong sample. Approximate:
    # effective = oracle * (1 - (1-v_acc)^n_correct_expected)
    # Simpler model: coverage * verifier_acc (conservative lower bound)
    if verifier_acc < 1.0:
        # More nuanced: among k samples, expected correct = k*p
        # Verifier sees all k, picks one. P(picks a correct one) given n_correct:
        # = 1 - P(picks wrong for all correct) ≈ verifier_acc for practical purposes
        adjusted_coverage = oracle_coverage * (1.0 - (1.0 - verifier_acc) ** np.clip(k_range * is_p, 0.1, None))
        adjusted_coverage = np.clip(adjusted_coverage, 0, 1)
    else:
        adjusted_coverage = oracle_coverage

    current_coverage = adjusted_coverage[is_k - 1]

    # ── Chart 1: Coverage vs k ──────────────────────────────────────────────
    fig_cov = go.Figure()

    if verifier_acc < 1.0:
        fig_cov.add_trace(go.Scatter(
            x=k_range, y=oracle_coverage,
            mode="lines", name="Oracle Verifier",
            line=dict(color=COLORS["gray"], width=2, dash="dash"),
        ))

    fig_cov.add_trace(go.Scatter(
        x=k_range, y=adjusted_coverage,
        mode="lines", name=f"Coverage ({verifier_type})",
        line=dict(color=COLORS["green"], width=3),
    ))

    # Current position marker
    fig_cov.add_trace(go.Scatter(
        x=[is_k], y=[current_coverage],
        mode="markers", name=f"k={is_k}",
        marker=dict(size=14, color=COLORS["yellow"], symbol="star",
                    line=dict(width=2, color=COLORS["white"])),
    ))

    fig_cov.update_layout(
        title="Coverage (pass@k) vs Number of Samples",
        xaxis_title="Number of Samples (k)",
        yaxis_title="Coverage (probability of at least one correct)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=450,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    # ── Metrics ─────────────────────────────────────────────────────────────
    # Find k needed for 50% and 90% coverage
    k_for_50 = None
    k_for_90 = None
    for ki in range(len(adjusted_coverage)):
        if adjusted_coverage[ki] >= 0.5 and k_for_50 is None:
            k_for_50 = ki + 1
        if adjusted_coverage[ki] >= 0.9 and k_for_90 is None:
            k_for_90 = ki + 1

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(f"Coverage at k={is_k}", f"{current_coverage:.1%}")
    with col_m2:
        st.metric("k for 50% Coverage", f"{k_for_50 if k_for_50 else '>256'}")
    with col_m3:
        st.metric("k for 90% Coverage", f"{k_for_90 if k_for_90 else '>256'}")

    # ── Sample grid visualization ───────────────────────────────────────────
    st.markdown("#### Sample Grid Visualization")
    st.markdown(f"Simulating **{is_k}** samples with **p={is_p:.2f}** success rate:")

    rng = np.random.RandomState(7)
    sample_correct = rng.rand(is_k) < is_p

    # Verifier picks
    if verifier_acc == 1.0:
        # Oracle picks any correct sample
        verifier_picks = sample_correct.copy()
    else:
        # Noisy verifier: for each sample, may flip the label
        verifier_picks = sample_correct.copy()
        for i in range(is_k):
            if rng.rand() > verifier_acc:
                verifier_picks[i] = not verifier_picks[i]

    # Build grid: up to 16 cols
    grid_cols = min(is_k, 16)
    grid_rows = (is_k + grid_cols - 1) // grid_cols

    grid_html = '<div style="display:grid; grid-template-columns: repeat(' + str(grid_cols) + ', 1fr); gap:4px; max-width:600px;">'
    for i in range(is_k):
        bg = COLORS["green"] if sample_correct[i] else COLORS["red"]
        border = f"3px solid {COLORS['yellow']}" if verifier_picks[i] else "2px solid #2A2D3E"
        opacity = "1.0" if sample_correct[i] else "0.6"
        grid_html += f'<div style="background:{bg}; opacity:{opacity}; border:{border}; border-radius:4px; height:28px; width:100%;"></div>'
    grid_html += '</div>'

    st.markdown(grid_html, unsafe_allow_html=True)
    st.markdown(f"""
<span style="color:{COLORS['gray']}; font-size:0.8rem;">
Green = correct sample, Red = incorrect, Yellow border = verifier's pick.
{int(sample_correct.sum())} of {is_k} samples correct ({sample_correct.mean():.0%}).
</span>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="insight-box">
<strong>Key insight:</strong> Even with a low per-sample success rate of {is_p:.0%}, generating {is_k} samples gives
<strong>{current_coverage:.1%}</strong> chance of at least one correct answer. The formula
pass@k = 1 &minus; (1&minus;p)<sup>k</sup> shows exponential improvement — but you need a <em>verifier</em>
to identify which sample is correct. The quality of the verifier is the bottleneck.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Compute Tradeoff
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Weak Model + Many Samples vs Strong Model + Few Samples")
    st.markdown("Given a fixed compute budget, is it better to run a cheap model many times or an expensive model a few times?")

    col_w, col_s = st.columns(2)
    with col_w:
        st.markdown(f"**Weak Model**")
        weak_p = st.slider("Success rate (p)", 0.01, 0.30, 0.05, step=0.01, key="is_weak_p")
        weak_cost = st.slider("Cost per sample", 1, 20, 2, key="is_weak_cost")
    with col_s:
        st.markdown(f"**Strong Model**")
        strong_p = st.slider("Success rate (p)", 0.10, 0.80, 0.30, step=0.05, key="is_strong_p")
        strong_cost = st.slider("Cost per sample", 5, 100, 20, key="is_strong_cost")

    budget_max = st.slider("Total Compute Budget", 10, 500, 200, step=10, key="is_budget")

    # ── Compute coverage vs budget ──────────────────────────────────────────
    budgets = np.arange(1, budget_max + 1)

    weak_coverage = []
    strong_coverage = []
    for b in budgets:
        k_weak = max(1, b // weak_cost)
        k_strong = max(1, b // strong_cost)
        cov_w = 1.0 - (1.0 - weak_p) ** k_weak
        cov_s = 1.0 - (1.0 - strong_p) ** k_strong
        weak_coverage.append(cov_w)
        strong_coverage.append(cov_s)

    weak_coverage = np.array(weak_coverage)
    strong_coverage = np.array(strong_coverage)

    # Find crossover point
    crossover = None
    for i in range(1, len(budgets)):
        if weak_coverage[i] >= strong_coverage[i] and weak_coverage[i - 1] < strong_coverage[i - 1]:
            crossover = budgets[i]
            break

    # ── Chart: Coverage vs Compute Budget ───────────────────────────────────
    fig_tradeoff = go.Figure()

    fig_tradeoff.add_trace(go.Scatter(
        x=budgets, y=weak_coverage,
        mode="lines", name=f"Weak (p={weak_p}, cost={weak_cost})",
        line=dict(color=COLORS["orange"], width=3),
    ))
    fig_tradeoff.add_trace(go.Scatter(
        x=budgets, y=strong_coverage,
        mode="lines", name=f"Strong (p={strong_p}, cost={strong_cost})",
        line=dict(color=COLORS["blue"], width=3),
    ))

    if crossover is not None:
        cross_idx = np.where(budgets == crossover)[0][0]
        fig_tradeoff.add_trace(go.Scatter(
            x=[crossover], y=[weak_coverage[cross_idx]],
            mode="markers", name=f"Crossover (budget={crossover})",
            marker=dict(size=14, color=COLORS["yellow"], symbol="diamond",
                        line=dict(width=2, color=COLORS["white"])),
        ))
        fig_tradeoff.add_vline(x=crossover, line_dash="dot", line_color=COLORS["yellow"], opacity=0.5)

    fig_tradeoff.update_layout(
        title="Coverage vs Compute Budget",
        xaxis_title="Compute Budget (units)",
        yaxis_title="Coverage (pass@k)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=480,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # ── Metrics ─────────────────────────────────────────────────────────────
    k_weak_at_max = max(1, budget_max // weak_cost)
    k_strong_at_max = max(1, budget_max // strong_cost)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric(f"Weak: {k_weak_at_max} samples", f"{weak_coverage[-1]:.1%} coverage")
    with col_t2:
        st.metric(f"Strong: {k_strong_at_max} samples", f"{strong_coverage[-1]:.1%} coverage")
    with col_t3:
        if crossover:
            st.metric("Crossover Budget", f"{crossover}")
        else:
            winner = "Weak" if weak_coverage[-1] > strong_coverage[-1] else "Strong"
            st.metric("Better at Max Budget", winner)

    # ── Insight ─────────────────────────────────────────────────────────────
    if crossover:
        insight_text = (
            f"The weak model overtakes the strong model at a compute budget of <strong>{crossover}</strong>. "
            f"Below that threshold, the strong model's higher per-sample accuracy wins. "
            f"Above it, the weak model's ability to generate many more samples (at {weak_cost}x cheaper) "
            f"gives it superior coverage through sheer volume &mdash; as long as you have a reliable verifier."
        )
    elif weak_coverage[-1] > strong_coverage[-1]:
        insight_text = (
            f"The weak model dominates across the entire budget range! At cost={weak_cost} per sample, "
            f"it generates {k_weak_at_max} samples vs the strong model's {k_strong_at_max}, achieving "
            f"<strong>{weak_coverage[-1]:.1%}</strong> vs <strong>{strong_coverage[-1]:.1%}</strong> coverage."
        )
    else:
        insight_text = (
            f"The strong model wins across the entire budget range. Its per-sample success rate of "
            f"{strong_p:.0%} is high enough that fewer expensive samples outperform many cheap ones. "
            f"Try lowering the strong model's success rate or increasing its cost to see a crossover."
        )

    st.markdown(f"""
<div class="insight-box">
<strong>Key insight:</strong> {insight_text}<br/><br/>
This is the core tradeoff of inference-time compute scaling: a weaker model sampled many times,
combined with a good verifier, can match or exceed a single call to a much stronger model.
This only works when verification is cheap and reliable &mdash; exactly the
<strong>judgment vs. verification</strong> axis from the flywheel demo.
</div>
""", unsafe_allow_html=True)
