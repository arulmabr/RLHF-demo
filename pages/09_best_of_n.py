"""
Page 9 -- Best-of-N / Rejection Sampling
The simplest "RL": generate N responses, pick the best one.
"""

from style import inject_custom_css, COLORS
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION IX</p>',
    unsafe_allow_html=True,
)
st.title("Best-of-N / Rejection Sampling")

st.markdown(
    """
    The simplest form of "RL from human feedback" requires **no gradient updates
    at all**.  The idea is almost embarrassingly straightforward:

    1. Sample **N** candidate responses from the policy.
    2. Score each one with a reward model.
    3. Return the highest-scoring response.
    """
)

st.markdown(
    '<div class="big-formula">'
    "y* = argmax<sub>i=1..N</sub> r(x, y<sub>i</sub>)"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    Despite its simplicity, Best-of-N is **competitive with PPO at low KL
    budgets**, requires no training loop, introduces no instability, and is
    trivial to implement.  The cost is purely at inference time: you must
    generate N completions instead of one, and the KL divergence from the base
    policy grows **logarithmically** with N.

    | N | KL cost (nats) |
    |---|----------------|
    | 4 | ~0.64 |
    | 16 | ~1.83 |
    | 64 | ~3.18 |
    | 128 | ~3.87 |
    """
)

st.markdown("---")

# =====================================================================
# 1. SAMPLING VISUALIZATION
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Best-of-N Sampling Visualization")
st.markdown(
    "Sample N responses from a standard normal distribution (representing "
    "response quality scores). The **best** sample is highlighted. As N grows, "
    "the expected maximum increases -- but only as sqrt(2 log N)."
)

col_n, col_seed = st.columns([3, 1])
with col_n:
    N = st.slider(
        "Number of samples (N)",
        min_value=1,
        max_value=128,
        value=16,
        step=1,
        key="bon_n_slider",
    )
with col_seed:
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        key="bon_seed",
    )

rng = np.random.RandomState(seed)
samples = rng.randn(N)
best_idx = int(np.argmax(samples))

# -- Dot plot of all samples with best highlighted --
marker_colors = [COLORS["blue"]] * N
marker_sizes = [8] * N
marker_colors[best_idx] = COLORS["red"]
marker_sizes[best_idx] = 18

fig_samples = go.Figure()

# All samples
fig_samples.add_trace(go.Scatter(
    x=list(range(N)),
    y=samples,
    mode="markers",
    marker=dict(
        color=marker_colors,
        size=marker_sizes,
        line=dict(width=1, color=COLORS["white"]),
    ),
    hovertemplate="Sample %{x}<br>Quality: %{y:.3f}<extra></extra>",
    name="Samples",
    showlegend=False,
))

# Highlight the best
fig_samples.add_trace(go.Scatter(
    x=[best_idx],
    y=[samples[best_idx]],
    mode="markers+text",
    marker=dict(color=COLORS["red"], size=20, symbol="star",
                line=dict(width=2, color=COLORS["yellow"])),
    text=[f"Best: {samples[best_idx]:.3f}"],
    textposition="top center",
    textfont=dict(color=COLORS["yellow"], size=13),
    name="Best-of-N",
    showlegend=False,
))

# Theoretical expected max line
expected_max = np.sqrt(2 * np.log(max(N, 2)))
fig_samples.add_hline(
    y=expected_max, line_dash="dash", line_color=COLORS["orange"],
    annotation_text=f"E[max] = sqrt(2 ln {N}) = {expected_max:.2f}",
    annotation_position="top left",
    annotation_font=dict(color=COLORS["orange"], size=12),
)

fig_samples.update_layout(
    title=f"Best-of-{N} Sampling (best = {samples[best_idx]:.3f})",
    xaxis_title="Sample index",
    yaxis_title="Quality score",
    height=400,
)
st.plotly_chart(fig_samples, use_container_width=True)

# -- Distribution of max across many trials --
st.markdown("##### Distribution of the Maximum Across 2000 Trials")

n_trials = 2000
trial_rng = np.random.RandomState(seed + 1)
all_maxes = np.array([trial_rng.randn(N).max() for _ in range(n_trials)])

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=all_maxes,
    nbinsx=50,
    marker_color=COLORS["blue"],
    opacity=0.75,
    name=f"max of {N} samples",
))

fig_dist.add_vline(
    x=expected_max, line_dash="dash", line_color=COLORS["orange"],
    annotation_text=f"E[max] = {expected_max:.2f}",
    annotation_position="top right",
    annotation_font=dict(color=COLORS["orange"], size=12),
)
fig_dist.add_vline(
    x=np.mean(all_maxes), line_dash="dot", line_color=COLORS["green"],
    annotation_text=f"Empirical mean = {np.mean(all_maxes):.2f}",
    annotation_position="top left",
    annotation_font=dict(color=COLORS["green"], size=12),
)

fig_dist.update_layout(
    title=f"Distribution of max(X_1, ..., X_{N}) Over {n_trials} Trials",
    xaxis_title="Maximum quality score",
    yaxis_title="Count",
    height=350,
    showlegend=True,
)
st.plotly_chart(fig_dist, use_container_width=True)

# -- Expected max vs N (theoretical curve) --
st.markdown("##### Expected Maximum vs N")

n_range = np.arange(1, 129)
expected_max_curve = np.array([
    np.sqrt(2 * np.log(max(n, 2))) for n in n_range
])

fig_emax = go.Figure()
fig_emax.add_trace(go.Scatter(
    x=n_range,
    y=expected_max_curve,
    mode="lines",
    line=dict(color=COLORS["cyan"], width=3),
    name="E[max] = sqrt(2 ln N)",
))

# Mark the current N
fig_emax.add_trace(go.Scatter(
    x=[N],
    y=[np.sqrt(2 * np.log(max(N, 2)))],
    mode="markers",
    marker=dict(color=COLORS["red"], size=14, symbol="diamond",
                line=dict(width=2, color=COLORS["white"])),
    name=f"Current N={N}",
))

fig_emax.update_layout(
    title="Expected Maximum Quality vs N (Gaussian Samples)",
    xaxis_title="N (number of samples)",
    yaxis_title="Expected max quality (std units)",
    height=350,
)
st.plotly_chart(fig_emax, use_container_width=True)

st.markdown("---")

# =====================================================================
# 2. KL COST CALCULATOR
# =====================================================================
st.markdown(
    '<p class="section-header">KL COST ANALYSIS</p>',
    unsafe_allow_html=True,
)
st.subheader("KL Divergence Cost of Best-of-N")
st.markdown(
    "Best-of-N implicitly defines a new policy that is shifted toward "
    "higher-reward responses. The KL divergence between this implicit policy "
    "and the base policy grows **logarithmically** with N:"
)

st.markdown(
    '<div class="big-formula">'
    "KL(pi_BoN || pi_base) = log(N) - (N-1)/N"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "For large N this is approximately **ln(N)**. Compare this with PPO, "
    "which can achieve arbitrary KL budgets through its penalty coefficient."
)

n_kl_range = np.arange(1, 257)
# Exact KL for Best-of-N: log(N) - (N-1)/N
kl_bon = np.log(n_kl_range) - (n_kl_range - 1) / n_kl_range
kl_bon[0] = 0.0  # KL for N=1 is 0

# Simulated PPO KL: grows linearly for the same quality improvement.
# PPO with a KL penalty can target any KL budget, but for equivalent
# quality improvement it typically uses less KL at higher budgets.
# We model PPO as achieving the same reward improvement at lower KL
# beyond a crossover point.

# Quality improvement for BoN: E[max of N gaussians]
quality_bon = np.array([
    np.sqrt(2 * np.log(max(n, 2))) if n > 1 else 0.0 for n in n_kl_range
])

# PPO: more efficient at high KL. Model as quality ~ sqrt(2 * KL) * efficiency_factor
# At low KL, PPO is similar; at high KL, PPO dominates.
ppo_efficiency = 1.15  # PPO gets 15% more reward per unit KL at high KL
kl_ppo_equivalent = (quality_bon / ppo_efficiency) ** 2 / 2.0

# Find crossover
crossover_idx = None
for i in range(len(n_kl_range)):
    if kl_bon[i] > kl_ppo_equivalent[i] and kl_bon[i] > 0.5:
        crossover_idx = i
        break

fig_kl = go.Figure()

fig_kl.add_trace(go.Scatter(
    x=n_kl_range,
    y=kl_bon,
    mode="lines",
    line=dict(color=COLORS["blue"], width=3),
    name="Best-of-N: KL = log(N) - (N-1)/N",
))

fig_kl.add_trace(go.Scatter(
    x=n_kl_range,
    y=kl_ppo_equivalent,
    mode="lines",
    line=dict(color=COLORS["red"], width=3, dash="dash"),
    name="PPO: KL for equivalent quality",
))

if crossover_idx is not None:
    fig_kl.add_trace(go.Scatter(
        x=[n_kl_range[crossover_idx]],
        y=[kl_bon[crossover_idx]],
        mode="markers+text",
        marker=dict(color=COLORS["yellow"], size=14, symbol="x",
                    line=dict(width=2, color=COLORS["white"])),
        text=[f"Crossover N={n_kl_range[crossover_idx]}"],
        textposition="top right",
        textfont=dict(color=COLORS["yellow"], size=12),
        name="Crossover point",
        showlegend=True,
    ))

fig_kl.update_layout(
    title="KL Cost: Best-of-N vs PPO (for equivalent quality)",
    xaxis_title="N (number of samples)",
    yaxis_title="KL divergence (nats)",
    height=420,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_kl, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Logarithmic growth:</strong> Doubling N adds only about 0.69 "
    "nats of KL. This means Best-of-N is remarkably cheap in terms of "
    "distributional shift -- but you pay at inference time with N forward "
    "passes. PPO amortizes this cost into training, eventually becoming more "
    "KL-efficient at higher quality targets."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. REWARD vs KL FRONTIER
# =====================================================================
st.markdown(
    '<p class="section-header">REWARD-KL TRADEOFF</p>',
    unsafe_allow_html=True,
)
st.subheader("Reward vs KL Frontier")
st.markdown(
    "The central question: **how much reward improvement do you get per unit "
    "of KL divergence?** Best-of-N and PPO trace different frontiers. "
    "Best-of-N is competitive at low KL budgets but PPO dominates at higher "
    "budgets because it can reshape the full distribution, not just select "
    "from samples."
)

# Build frontier points for various N values
n_frontier = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
kl_frontier_bon = np.log(n_frontier) - (n_frontier - 1) / n_frontier
kl_frontier_bon[0] = 0.0
reward_frontier_bon = np.array([
    np.sqrt(2 * np.log(max(n, 2))) if n > 1 else 0.0 for n in n_frontier
])

# PPO frontier: same KL range but higher reward at high KL
kl_range_ppo = np.linspace(0, 6.0, 200)
# PPO reward model: concave, starts similar to BoN, then pulls ahead
# Use a model: reward_ppo = alpha * sqrt(KL) + beta * KL^0.3
# Calibrate so it matches BoN at low KL and exceeds at high KL
reward_ppo = 0.9 * np.sqrt(2 * kl_range_ppo) + 0.15 * kl_range_ppo ** 0.4
reward_ppo[0] = 0.0

fig_frontier = go.Figure()

# PPO frontier (shaded area)
fig_frontier.add_trace(go.Scatter(
    x=kl_range_ppo,
    y=reward_ppo,
    mode="lines",
    line=dict(color=COLORS["red"], width=3),
    name="PPO frontier",
    fill=None,
))

# Best-of-N frontier
fig_frontier.add_trace(go.Scatter(
    x=kl_frontier_bon,
    y=reward_frontier_bon,
    mode="lines+markers",
    line=dict(color=COLORS["blue"], width=3),
    marker=dict(size=7, color=COLORS["blue"],
                line=dict(width=1, color=COLORS["white"])),
    name="Best-of-N frontier",
))

# Label key N values
for n_val, kl_val, r_val in zip(n_frontier, kl_frontier_bon, reward_frontier_bon):
    if n_val in [4, 16, 64, 256]:
        fig_frontier.add_annotation(
            x=kl_val, y=r_val,
            text=f"N={n_val}",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor=COLORS["cyan"],
            font=dict(color=COLORS["cyan"], size=11),
            ax=25, ay=-25,
        )

fig_frontier.update_layout(
    title="Reward Improvement vs KL Cost",
    xaxis_title="KL divergence from base policy (nats)",
    yaxis_title="Expected reward improvement (std units)",
    height=450,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_frontier, use_container_width=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#4A90D9;">Best-of-N strengths</strong>'
        "<ul>"
        "<li>No training required</li>"
        "<li>No instability or mode collapse</li>"
        "<li>Competitive at low KL (N &le; 16)</li>"
        "<li>Easy to swap reward models</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
with col_f2:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#E74C3C;">PPO strengths</strong>'
        "<ul>"
        "<li>Amortizes cost into training</li>"
        "<li>More KL-efficient at high budgets</li>"
        "<li>Reshapes entire distribution</li>"
        "<li>Single forward pass at inference</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 4. ITERATED REJECTION SAMPLING
# =====================================================================
st.markdown(
    '<p class="section-header">ADVANCED</p>',
    unsafe_allow_html=True,
)
st.subheader("Iterated Rejection Sampling (Iterated Best-of-N)")
st.markdown(
    """
    A natural extension: **repeat** the Best-of-N process iteratively.
    Each round, generate N samples from the current model, select the best,
    and fine-tune (SFT) on the selected responses. This creates a sequence
    of progressively better models:

    - **Round 1:** Generate N from base model, select best, SFT on selected
    - **Round 2:** Generate N from improved model, select best, SFT on selected
    - **Round 3:** Continue...

    Each round shifts the distribution further toward high reward, but also
    **compounds reward model errors** -- the model may overfit to the reward
    model's blind spots.
    """
)

col_iter_n, col_iter_noise = st.columns(2)
with col_iter_n:
    iter_N = st.slider(
        "Samples per round (N)",
        min_value=2,
        max_value=64,
        value=16,
        step=1,
        key="iter_n_slider",
    )
with col_iter_noise:
    rm_noise = st.slider(
        "Reward model noise (sigma)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="rm_noise_slider",
        help="Higher values simulate a noisier reward model, amplifying error compounding.",
    )

n_rounds = 5
iter_rng = np.random.RandomState(seed + 100)

# Simulate iterated rejection sampling
# We track:
#   - mean of the true quality distribution each round
#   - mean of the reward-model-perceived quality
#   - standard deviation (representing diversity)
true_mean = 0.0
true_std = 1.0
round_true_means = [true_mean]
round_perceived_means = [true_mean]
round_stds = [true_std]
round_selected_true = []
round_selected_perceived = []

for r in range(n_rounds):
    # Generate N samples from current distribution
    true_samples = iter_rng.normal(true_mean, true_std, iter_N)
    # Reward model adds noise
    perceived_scores = true_samples + iter_rng.normal(0, rm_noise, iter_N)
    # Select best according to reward model
    best = int(np.argmax(perceived_scores))
    selected_true = true_samples[best]
    selected_perceived = perceived_scores[best]

    round_selected_true.append(selected_true)
    round_selected_perceived.append(selected_perceived)

    # SFT on selected: shift distribution toward selected value
    # Model the SFT as moving the mean toward the selected sample
    shift_rate = 0.6  # how much the mean shifts toward the selected
    true_mean = true_mean + shift_rate * (selected_true - true_mean)
    # Diversity decreases slightly each round
    true_std = max(0.3, true_std * 0.92)

    round_true_means.append(true_mean)
    round_perceived_means.append(
        true_mean + rm_noise * np.sqrt(2 * np.log(max(iter_N, 2)))
    )
    round_stds.append(true_std)

rounds_x = list(range(n_rounds + 1))
round_labels = ["Base"] + [f"Round {i+1}" for i in range(n_rounds)]

fig_iter = go.Figure()

# True quality mean trajectory
fig_iter.add_trace(go.Scatter(
    x=rounds_x,
    y=round_true_means,
    mode="lines+markers",
    line=dict(color=COLORS["green"], width=3),
    marker=dict(size=10, color=COLORS["green"],
                line=dict(width=1, color=COLORS["white"])),
    name="True quality (mean)",
))

# Perceived quality trajectory
fig_iter.add_trace(go.Scatter(
    x=rounds_x,
    y=round_perceived_means,
    mode="lines+markers",
    line=dict(color=COLORS["orange"], width=3, dash="dash"),
    marker=dict(size=10, color=COLORS["orange"],
                line=dict(width=1, color=COLORS["white"])),
    name="RM-perceived quality (mean)",
))

# Diversity band (true_mean +/- true_std)
upper_band = [m + s for m, s in zip(round_true_means, round_stds)]
lower_band = [m - s for m, s in zip(round_true_means, round_stds)]

fig_iter.add_trace(go.Scatter(
    x=rounds_x + rounds_x[::-1],
    y=upper_band + lower_band[::-1],
    fill="toself",
    fillcolor="rgba(74, 144, 217, 0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Diversity (1 std)",
    showlegend=True,
))

fig_iter.update_layout(
    title=f"Iterated Rejection Sampling (N={iter_N}, RM noise={rm_noise:.2f})",
    xaxis=dict(
        title="Iteration",
        tickvals=rounds_x,
        ticktext=round_labels,
    ),
    yaxis_title="Quality (std units)",
    height=420,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_iter, use_container_width=True)

# Per-round detail table
st.markdown("##### Per-Round Details")

detail_cols = st.columns(n_rounds)
for i in range(n_rounds):
    with detail_cols[i]:
        gap = round_selected_perceived[i] - round_selected_true[i]
        st.markdown(
            f'<div class="concept-card" style="text-align:center;">'
            f'<strong style="color:{COLORS["cyan"]};">Round {i+1}</strong><br>'
            f'True quality: <span style="color:{COLORS["green"]};">'
            f'{round_selected_true[i]:.2f}</span><br>'
            f'RM perceived: <span style="color:{COLORS["orange"]};">'
            f'{round_selected_perceived[i]:.2f}</span><br>'
            f'RM overestimate: <span style="color:{COLORS["red"]};">'
            f'{gap:+.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Show distributions shifting across rounds
st.markdown("##### How the Quality Distribution Shifts Each Round")

fig_dists = go.Figure()
x_range = np.linspace(-3, 5, 300)

color_scale = [COLORS["blue"], COLORS["cyan"], COLORS["green"],
               COLORS["yellow"], COLORS["orange"], COLORS["red"]]

for i in range(n_rounds + 1):
    mu = round_true_means[i]
    sigma = round_stds[i]
    y_vals = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range - mu) / sigma) ** 2
    )
    label = "Base model" if i == 0 else f"After round {i}"
    fig_dists.add_trace(go.Scatter(
        x=x_range,
        y=y_vals,
        mode="lines",
        line=dict(color=color_scale[i % len(color_scale)], width=2.5),
        name=label,
    ))

fig_dists.update_layout(
    title="Quality Distribution Shifting Across Rounds",
    xaxis_title="Quality score",
    yaxis_title="Density",
    height=380,
    legend=dict(x=0.72, y=0.98),
)
st.plotly_chart(fig_dists, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Compounding errors:</strong> Notice the growing gap between "
    "true quality and RM-perceived quality. Each round the model shifts "
    "toward what the reward model <em>thinks</em> is good, which may diverge "
    "from what is <em>actually</em> good. With a noisy RM (try increasing "
    "the noise slider), the gap widens rapidly -- this is reward hacking "
    "via iterated amplification of RM errors."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 5. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.05rem; padding:20px 24px;">'
    "<strong>Key Insight:</strong> Best-of-N is underrated. Trivial to "
    "implement, no training instability, competitive with PPO at low KL "
    "budgets. The KL cost grows only logarithmically: doubling N adds "
    "just ~0.69 nats. For many practical applications -- especially when "
    "you need to swap reward models frequently or want a strong baseline "
    "before committing to full RL training -- Best-of-N is the right "
    "starting point."
    "</div>",
    unsafe_allow_html=True,
)
