import streamlit as st

st.set_page_config(page_title="Model Collapse Explorer", page_icon="📉", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">DATA FLYWHEELS &amp; SELF-IMPROVING SYSTEMS</p>', unsafe_allow_html=True)
st.title("Model Collapse Explorer")
st.markdown("Watch what happens when models are trained on their own outputs — distributions narrow, tails vanish, and diversity collapses.")
st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Collapse Visualizer", "Mitigation Lab"])

# ── Shared generation logic ─────────────────────────────────────────────────
def generate_initial_samples(dist_type, n_samples, rng):
    if dist_type == "Gaussian":
        return rng.randn(n_samples) * 2.0
    elif dist_type == "Bimodal":
        mix = rng.rand(n_samples) < 0.5
        return np.where(mix, rng.randn(n_samples) * 0.8 - 2.5, rng.randn(n_samples) * 0.8 + 2.5)
    else:  # Heavy-Tailed
        return rng.standard_t(df=3, size=n_samples) * 1.5


def collapse_iterate(samples, n_generations, n_samples, rng, synthetic_ratio=1.0):
    """Run iterative model collapse. synthetic_ratio=1.0 means pure self-training."""
    original = samples.copy()
    history = [samples.copy()]
    stds = [np.std(samples)]
    kl_divs = [0.0]

    for gen in range(n_generations):
        # Fit Gaussian to current samples (the "model")
        mu = np.mean(samples)
        sigma = max(np.std(samples), 0.01)

        # Resample from fitted Gaussian (synthetic data)
        n_synthetic = int(n_samples * synthetic_ratio)
        n_real = n_samples - n_synthetic
        synthetic = rng.randn(n_synthetic) * sigma + mu

        if n_real > 0:
            real_idx = rng.choice(len(original), size=n_real, replace=True)
            real = original[real_idx]
            samples = np.concatenate([synthetic, real])
        else:
            samples = synthetic

        history.append(samples.copy())
        stds.append(np.std(samples))

        # Approximate KL divergence via histogram comparison
        bins = np.linspace(-8, 8, 60)
        hist_orig, _ = np.histogram(original, bins=bins, density=True)
        hist_curr, _ = np.histogram(samples, bins=bins, density=True)
        hist_orig = np.clip(hist_orig, 1e-10, None)
        hist_curr = np.clip(hist_curr, 1e-10, None)
        # Normalize
        hist_orig = hist_orig / hist_orig.sum()
        hist_curr = hist_curr / hist_curr.sum()
        kl = np.sum(hist_orig * np.log(hist_orig / hist_curr))
        kl_divs.append(kl)

    return history, stds, kl_divs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Collapse Visualizer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        mc_generations = st.slider("Generations", 1, 15, 8, key="mc_gen")
    with col_ctrl2:
        mc_samples = st.slider("Samples per Generation", 20, 500, 200, step=20, key="mc_samp")
    with col_ctrl3:
        mc_dist = st.selectbox("Distribution Type", ["Gaussian", "Bimodal", "Heavy-Tailed"], key="mc_dist")
    with col_ctrl4:
        mc_seed = st.button("Re-roll", key="mc_reroll")

    # Seed management
    if "mc_seed_val" not in st.session_state:
        st.session_state.mc_seed_val = 42
    if mc_seed:
        st.session_state.mc_seed_val = np.random.randint(0, 100000)

    rng = np.random.RandomState(st.session_state.mc_seed_val)
    initial_samples = generate_initial_samples(mc_dist, mc_samples, rng)
    history, stds, kl_divs = collapse_iterate(initial_samples, mc_generations, mc_samples, rng)

    # ── Chart 1: Overlaid density curves ────────────────────────────────────
    fig_density = go.Figure()
    x_range = np.linspace(-8, 8, 300)

    for gen_i in range(len(history)):
        # Blue (early) to red (late) gradient
        t = gen_i / max(len(history) - 1, 1)
        r = int(74 + t * (231 - 74))
        g = int(144 - t * (144 - 76))
        b = int(217 - t * (217 - 60))
        color = f"rgb({r},{g},{b})"

        samples = history[gen_i]
        mu, sigma = np.mean(samples), max(np.std(samples), 0.01)
        density = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        fig_density.add_trace(go.Scatter(
            x=x_range, y=density,
            mode="lines", name=f"Gen {gen_i}",
            line=dict(color=color, width=2 if gen_i == 0 or gen_i == len(history) - 1 else 1),
            opacity=0.9 if gen_i == 0 or gen_i == len(history) - 1 else 0.5,
        ))

    fig_density.update_layout(
        title="Distribution Across Generations (Blue→Red = Early→Late)",
        xaxis_title="Value",
        yaxis_title="Density",
        height=450,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # ── Chart 2: Std Dev and KL Divergence ──────────────────────────────────
    gen_x = list(range(len(stds)))

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(
        x=gen_x, y=stds,
        mode="lines+markers", name="Std Dev",
        line=dict(color=COLORS["blue"], width=3),
        marker=dict(size=7),
    ))
    fig_metrics.add_trace(go.Scatter(
        x=gen_x, y=kl_divs,
        mode="lines+markers", name="KL Divergence from Original",
        line=dict(color=COLORS["red"], width=3),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig_metrics.update_layout(
        title="Collapse Metrics Over Generations",
        xaxis_title="Generation",
        yaxis=dict(title="Standard Deviation", titlefont=dict(color=COLORS["blue"]),
                   tickfont=dict(color=COLORS["blue"])),
        yaxis2=dict(title="KL Divergence", titlefont=dict(color=COLORS["red"]),
                    tickfont=dict(color=COLORS["red"]),
                    overlaying="y", side="right"),
        height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown(f"""
<div class="insight-box">
<strong>What you're seeing:</strong> Each generation fits a Gaussian to the current data and resamples from it.
This is analogous to training a model on its own outputs. The tails of the distribution &mdash; rare but
important examples &mdash; are lost first. After a few generations, the distribution collapses to a narrow
spike around the mean. Std dev drops from <strong>{stds[0]:.2f}</strong> to <strong>{stds[-1]:.2f}</strong>
and KL divergence from the original reaches <strong>{kl_divs[-1]:.3f}</strong>.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Mitigation Lab
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Can We Prevent Collapse?")
    st.markdown("Mix in real (original) data with synthetic data each generation and see if collapse slows or stops.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        mit_generations = st.slider("Generations", 1, 15, 10, key="mit_gen")
    with col_m2:
        mit_samples = st.slider("Samples per Generation", 20, 500, 200, step=20, key="mit_samp")
    with col_m3:
        mit_dist = st.selectbox("Distribution Type", ["Gaussian", "Bimodal", "Heavy-Tailed"], key="mit_dist")
    with col_m4:
        synthetic_pct = st.slider("Synthetic Data %", 0, 100, 80, step=5, key="mit_ratio",
                                  help="Percentage of each generation's data that is synthetic (model-generated). The rest is real data.")

    # Use same seed for fair comparison
    rng_pure = np.random.RandomState(101)
    rng_mixed = np.random.RandomState(101)

    initial_pure = generate_initial_samples(mit_dist, mit_samples, rng_pure)
    initial_mixed = initial_pure.copy()

    # Pure self-training (100% synthetic)
    _, stds_pure, kl_pure = collapse_iterate(
        initial_pure, mit_generations, mit_samples, np.random.RandomState(101), synthetic_ratio=1.0
    )
    # Mixed training
    _, stds_mixed, kl_mixed = collapse_iterate(
        initial_mixed, mit_generations, mit_samples, np.random.RandomState(101), synthetic_ratio=synthetic_pct / 100.0
    )

    # ── Side by side comparison ─────────────────────────────────────────────
    gen_x = list(range(len(stds_pure)))

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=gen_x, y=stds_pure,
        mode="lines+markers", name="Pure Self-Training (100% synthetic)",
        line=dict(color=COLORS["red"], width=3),
        marker=dict(size=7),
    ))
    fig_compare.add_trace(go.Scatter(
        x=gen_x, y=stds_mixed,
        mode="lines+markers", name=f"Mixed Training ({synthetic_pct}% synthetic)",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=7),
    ))
    fig_compare.add_hline(
        y=stds_pure[0], line_dash="dot", line_color=COLORS["gray"],
        annotation_text="Original Std Dev", annotation_position="top right",
    )
    fig_compare.update_layout(
        title="Standard Deviation: Pure vs Mixed Training",
        xaxis_title="Generation",
        yaxis_title="Standard Deviation",
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── KL comparison ───────────────────────────────────────────────────────
    fig_kl = go.Figure()
    fig_kl.add_trace(go.Scatter(
        x=gen_x, y=kl_pure,
        mode="lines+markers", name="Pure Self-Training",
        line=dict(color=COLORS["red"], width=3),
        marker=dict(size=7),
    ))
    fig_kl.add_trace(go.Scatter(
        x=gen_x, y=kl_mixed,
        mode="lines+markers", name=f"Mixed ({synthetic_pct}% synthetic)",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=7),
    ))
    fig_kl.update_layout(
        title="KL Divergence from Original Distribution",
        xaxis_title="Generation",
        yaxis_title="KL Divergence",
        height=350,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_kl, use_container_width=True)

    # ── Metrics ─────────────────────────────────────────────────────────────
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Final Std Dev (Pure)", f"{stds_pure[-1]:.3f}")
    with col_r2:
        st.metric("Final Std Dev (Mixed)", f"{stds_mixed[-1]:.3f}")
    with col_r3:
        preservation = stds_mixed[-1] / max(stds_pure[0], 0.01) * 100
        st.metric("Diversity Preserved (Mixed)", f"{preservation:.0f}%")

    real_pct = 100 - synthetic_pct
    st.markdown(f"""
<div class="insight-box">
<strong>Key insight:</strong> Mixing in even a small fraction of real data ({real_pct}% here) dramatically
slows collapse. With pure self-training, std dev drops to <strong>{stds_pure[-1]:.3f}</strong> after
{mit_generations} generations. With {real_pct}% real data mixed in, it stays at
<strong>{stds_mixed[-1]:.3f}</strong>. This is why companies maintaining data flywheels must
carefully preserve access to original human-generated data &mdash; the "anchor" that prevents
the distribution from collapsing.
</div>
""", unsafe_allow_html=True)
