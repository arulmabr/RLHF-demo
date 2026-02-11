from style import inject_custom_css, COLORS, sigmoid
import streamlit as st
import numpy as np
import plotly.graph_objects as go

inject_custom_css()

st.markdown('<p class="section-header">SECTION V</p>', unsafe_allow_html=True)
st.title("Reward Modeling")
st.markdown("#### Compressing human preferences into a scalar")

st.markdown("---")

# ── 1. Explanation ────────────────────────────────────────────────────────────

st.markdown("""
<div class="concept-card">
<strong>Core Idea:</strong> Humans are better at <em>comparing</em> two outputs than
<em>generating</em> the ideal output from scratch. Reward modeling exploits this by
collecting pairwise comparisons and training a neural network to predict a scalar
reward <code>r(x, y)</code> that explains the observed preferences.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-formula">
Bradley-Terry model: &nbsp; P(y_w &#x227B; y_l | x) = &sigma;( r(x, y_w) - r(x, y_l) )
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-formula">
Loss: &nbsp; L_RM = -E[ log &sigma;( r(y_w) - r(y_l) ) ]
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── 2. "Be the Reward Model" Game ────────────────────────────────────────────

st.markdown("---")
st.markdown("### Be the Reward Model")
st.markdown(
    "You are the reward model. For each prompt, pick the response you prefer. "
    "After all pairs, we derive implicit reward scores from your choices using "
    "the Bradley-Terry model and compare you with a simulated average human."
)

PAIRS = [
    {
        "prompt": "Explain quantum entanglement to a 10-year-old.",
        "a": "Quantum entanglement is a phenomenon in quantum mechanics whereby "
             "two particles become correlated such that the quantum state of one "
             "instantaneously influences the other, irrespective of distance.",
        "b": "Imagine you have two magic coins. When you flip one and get heads, "
             "the other always lands tails -- no matter how far apart they are! "
             "Scientists call this quantum entanglement.",
        "human_prefers": "B",
    },
    {
        "prompt": "What is 15% of 200?",
        "a": "15% of 200 is 30.",
        "b": "Great question! So 15% means 15 out of every 100. Since 200 is "
             "two 100s, we take 15 twice, which gives us 30. The answer is 30! "
             "I hope this helps!",
        "human_prefers": "A",
    },
    {
        "prompt": "Write a haiku about rain.",
        "a": "Drops tap the window\nPuddles form on quiet streets\nEarth drinks and exhales",
        "b": "Rain falls from the sky\nIt makes everything all wet\nI don't like the rain",
        "human_prefers": "A",
    },
    {
        "prompt": "Should I invest in Bitcoin?",
        "a": "Absolutely! Bitcoin is the future of finance and has shown amazing "
             "returns. You should put as much money as you can into it right now.",
        "b": "I can share some considerations but cannot give financial advice. "
             "Bitcoin is volatile -- it has seen large gains and large losses. "
             "Factors to consider include your risk tolerance, time horizon, and "
             "portfolio diversification. Consulting a financial advisor is wise.",
        "human_prefers": "B",
    },
    {
        "prompt": "Summarize the theory of evolution in one sentence.",
        "a": "Species change over time through natural selection, where traits "
             "that improve survival and reproduction become more common across "
             "generations.",
        "b": "Evolution is like, things change and stuff, survival of the fittest "
             "basically means the strong ones live and the weak ones don't, so "
             "yeah, that's evolution.",
        "human_prefers": "A",
    },
    {
        "prompt": "How do I fix a leaky faucet?",
        "a": "First, turn off the water supply under the sink. Then remove the "
             "faucet handle (usually a screw under the cap). Replace the worn "
             "washer or O-ring inside, reassemble, and turn the water back on.",
        "b": "You should call a plumber. Plumbing is complicated and you could "
             "make it worse if you try to fix it yourself.",
        "human_prefers": "A",
    },
]

# Initialize session state
if "rm_choices" not in st.session_state:
    st.session_state.rm_choices = {}
if "rm_submitted" not in st.session_state:
    st.session_state.rm_submitted = False

# Response names for the items
RESPONSE_NAMES = ["A", "B"]

for i, pair in enumerate(PAIRS):
    st.markdown(f"**Pair {i + 1}: ** _{pair['prompt']}_")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""<div class="concept-card"><strong>Response A</strong><br/>{pair['a']}</div>""",
                    unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="concept-card"><strong>Response B</strong><br/>{pair['b']}</div>""",
                    unsafe_allow_html=True)

    choice = st.radio(
        f"Which response do you prefer? (Pair {i + 1})",
        options=["A", "B"],
        key=f"pair_{i}",
        horizontal=True,
    )
    st.session_state.rm_choices[i] = choice
    st.markdown("")

if st.button("Submit My Preferences", type="primary"):
    st.session_state.rm_submitted = True

if st.session_state.rm_submitted and len(st.session_state.rm_choices) == len(PAIRS):
    st.markdown("#### Results: Implicit Reward Scores")

    # Collect all unique items (each pair has item A and B)
    # Use simple Bradley-Terry MLE via iterative algorithm
    n_items = len(PAIRS) * 2  # A and B for each pair
    wins = np.zeros(n_items)
    comparisons = np.zeros((n_items, n_items))

    user_choices = []
    human_choices = []

    for i, pair in enumerate(PAIRS):
        idx_a = 2 * i
        idx_b = 2 * i + 1
        user_choice = st.session_state.rm_choices[i]
        user_choices.append(user_choice)
        human_choices.append(pair["human_prefers"])

        if user_choice == "A":
            wins[idx_a] += 1
            comparisons[idx_a, idx_b] += 1
        else:
            wins[idx_b] += 1
            comparisons[idx_b, idx_a] += 1

    # Simple score: log-odds from win counts + smoothing
    scores = np.zeros(n_items)
    for i in range(len(PAIRS)):
        idx_a = 2 * i
        idx_b = 2 * i + 1
        if st.session_state.rm_choices[i] == "A":
            scores[idx_a] = 1.0
            scores[idx_b] = -1.0
        else:
            scores[idx_a] = -1.0
            scores[idx_b] = 1.0

    # Compute human scores
    human_scores = np.zeros(n_items)
    for i, pair in enumerate(PAIRS):
        idx_a = 2 * i
        idx_b = 2 * i + 1
        if pair["human_prefers"] == "A":
            human_scores[idx_a] = 1.0
            human_scores[idx_b] = -1.0
        else:
            human_scores[idx_a] = -1.0
            human_scores[idx_b] = 1.0

    # Agreement
    agreement = sum(1 for u, h in zip(user_choices, human_choices) if u == h)
    agreement_rate = agreement / len(PAIRS)

    col_score, col_agree = st.columns([2, 1])

    with col_score:
        labels = []
        user_vals = []
        human_vals = []
        for i, pair in enumerate(PAIRS):
            chosen_label = f"P{i+1}: {'A' if st.session_state.rm_choices[i] == 'A' else 'B'} (chosen)"
            rejected_label = f"P{i+1}: {'B' if st.session_state.rm_choices[i] == 'A' else 'A'} (rejected)"
            labels.extend([chosen_label, rejected_label])
            if st.session_state.rm_choices[i] == "A":
                user_vals.extend([scores[2*i], scores[2*i+1]])
                human_vals.extend([human_scores[2*i], human_scores[2*i+1]])
            else:
                user_vals.extend([scores[2*i+1], scores[2*i]])
                human_vals.extend([human_scores[2*i+1], human_scores[2*i]])

        fig_scores = go.Figure()
        fig_scores.add_trace(go.Bar(
            name="Your Reward",
            x=labels,
            y=user_vals,
            marker_color=[COLORS["green"] if v > 0 else COLORS["red"] for v in user_vals],
            opacity=0.85,
        ))
        fig_scores.update_layout(
            title="Implicit Reward Scores from Your Choices",
            yaxis_title="Reward r(x, y)",
            xaxis_tickangle=-40,
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    with col_agree:
        st.markdown("")
        st.markdown("")
        color = COLORS["green"] if agreement_rate >= 0.7 else (COLORS["orange"] if agreement_rate >= 0.5 else COLORS["red"])
        st.markdown(f"""
<div class="concept-card" style="text-align:center; padding:24px;">
<span style="font-size:2.5rem; font-weight:700; color:{color};">{agreement_rate:.0%}</span><br/>
<span style="font-size:1rem; color:{COLORS['gray']};">Agreement with<br/>Average Human</span><br/><br/>
<span style="font-size:0.9rem; color:{COLORS['white']};">{agreement}/{len(PAIRS)} pairs matched</span>
</div>
""", unsafe_allow_html=True)

        for i in range(len(PAIRS)):
            match = "match" if user_choices[i] == human_choices[i] else "differ"
            icon = "+" if match == "match" else "-"
            st.markdown(
                f"Pair {i+1}: You chose **{user_choices[i]}**, human chose **{human_choices[i]}** "
                f"({'match' if match == 'match' else 'differ'})"
            )


# ── 3. Bradley-Terry Visualizer ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### Bradley-Terry Visualizer")
st.markdown(
    "Adjust the reward scores for the preferred (w) and dispreferred (l) "
    "responses. See how the probability, loss, and gradients change."
)

col_rw, col_rl = st.columns(2)
with col_rw:
    r_w = st.slider("r(y_w) -- reward for preferred response", -3.0, 3.0, 1.5, 0.1)
with col_rl:
    r_l = st.slider("r(y_l) -- reward for dispreferred response", -3.0, 3.0, -0.5, 0.1)

diff = r_w - r_l
prob_w = sigmoid(diff)
loss_val = -np.log(np.clip(prob_w, 1e-10, 1.0))

# Gradients of L_RM with respect to r_w and r_l
# dL/dr_w = -(1 - sigma(diff)) = sigma(diff) - 1
# dL/dr_l = (1 - sigma(diff)) = 1 - sigma(diff)
grad_rw = prob_w - 1.0  # negative: pushes r_w up (gradient descent decreases loss)
grad_rl = 1.0 - prob_w   # positive: pushes r_l down

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">P(y_w preferred)</span><br/>
<span style="font-size:2rem;font-weight:700;color:{COLORS['blue']};">{prob_w:.4f}</span><br/>
<span style="color:{COLORS['gray']};font-size:0.8rem;">&sigma;({diff:.2f})</span>
</div>
""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">Loss L_RM</span><br/>
<span style="font-size:2rem;font-weight:700;color:{COLORS['red']};">{loss_val:.4f}</span><br/>
<span style="color:{COLORS['gray']};font-size:0.8rem;">-log &sigma;({diff:.2f})</span>
</div>
""", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">Gradients (descent direction)</span><br/>
<span style="font-size:1.1rem;color:{COLORS['green']};">dL/dr_w = {grad_rw:.4f}</span><br/>
<span style="font-size:1.1rem;color:{COLORS['red']};">dL/dr_l = {grad_rl:+.4f}</span><br/>
<span style="color:{COLORS['gray']};font-size:0.75rem;">Descent pushes r_w up, r_l down</span>
</div>
""", unsafe_allow_html=True)

# Plot: P(y_w wins) and Loss as function of r_w - r_l, with current point marked
diffs = np.linspace(-6, 6, 300)
probs = sigmoid(diffs)
losses = -np.log(np.clip(probs, 1e-10, 1.0))

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(
    x=diffs, y=probs, name="P(y_w wins)",
    line=dict(color=COLORS["blue"], width=2.5),
))
fig_bt.add_trace(go.Scatter(
    x=diffs, y=losses, name="Loss",
    line=dict(color=COLORS["red"], width=2.5, dash="dash"),
))
# Current point
fig_bt.add_trace(go.Scatter(
    x=[diff], y=[prob_w], mode="markers",
    marker=dict(size=14, color=COLORS["blue"], symbol="circle",
                line=dict(width=2, color=COLORS["white"])),
    name=f"Current P = {prob_w:.3f}",
))
fig_bt.add_trace(go.Scatter(
    x=[diff], y=[loss_val], mode="markers",
    marker=dict(size=14, color=COLORS["red"], symbol="diamond",
                line=dict(width=2, color=COLORS["white"])),
    name=f"Current Loss = {loss_val:.3f}",
))
# Gradient arrows (annotation)
fig_bt.add_annotation(
    x=diff, y=loss_val,
    ax=diff + 1.2, ay=loss_val,
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=3, arrowsize=1.5,
    arrowcolor=COLORS["green"], arrowwidth=2,
    text="push r_w up",
    font=dict(color=COLORS["green"], size=11),
)
fig_bt.update_layout(
    title="Bradley-Terry: Probability & Loss vs Reward Difference",
    xaxis_title="r(y_w) - r(y_l)",
    yaxis_title="Value",
    height=450,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
)
st.plotly_chart(fig_bt, use_container_width=True)


# ── 4. Bias Sliders ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Reward Model Biases")
st.markdown(
    "Real reward models learn spurious correlations from human preferences. "
    "Adjust bias weights to see how they distort the reward signal."
)

st.markdown("""
<div class="big-formula">
total_reward = true_quality + length_bias &times; length + sycophancy_bias &times; agreement
+ format_bias &times; formatting + confidence_bias &times; confidence
</div>
""", unsafe_allow_html=True)

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    length_bias = st.slider("Length Bias", 0.0, 1.0, 0.0, 0.05, key="bias_len")
with col_b2:
    sycophancy_bias = st.slider("Sycophancy Bias", 0.0, 1.0, 0.0, 0.05, key="bias_syc")
with col_b3:
    format_bias = st.slider("Format Bias", 0.0, 1.0, 0.0, 0.05, key="bias_fmt")
with col_b4:
    confidence_bias = st.slider("Confidence Bias", 0.0, 1.0, 0.0, 0.05, key="bias_conf")

# Example responses with feature scores
EXAMPLE_RESPONSES = [
    {
        "label": "Concise correct answer",
        "true_quality": 0.9,
        "length": 0.1,
        "agreement": 0.3,
        "formatting": 0.2,
        "confidence": 0.5,
    },
    {
        "label": "Verbose correct answer",
        "true_quality": 0.8,
        "length": 0.9,
        "agreement": 0.3,
        "formatting": 0.7,
        "confidence": 0.6,
    },
    {
        "label": "Flattering but wrong",
        "true_quality": 0.2,
        "length": 0.5,
        "agreement": 1.0,
        "formatting": 0.5,
        "confidence": 0.9,
    },
    {
        "label": "Honest disagreement",
        "true_quality": 0.85,
        "length": 0.4,
        "agreement": 0.0,
        "formatting": 0.3,
        "confidence": 0.4,
    },
    {
        "label": "Well-formatted mediocre",
        "true_quality": 0.4,
        "length": 0.6,
        "agreement": 0.5,
        "formatting": 1.0,
        "confidence": 0.7,
    },
    {
        "label": "Overconfident nonsense",
        "true_quality": 0.1,
        "length": 0.7,
        "agreement": 0.6,
        "formatting": 0.4,
        "confidence": 1.0,
    },
]

# Compute total rewards
labels = []
true_quals = []
total_rewards = []
bias_contributions = []

for resp in EXAMPLE_RESPONSES:
    tq = resp["true_quality"]
    bias_part = (
        length_bias * resp["length"]
        + sycophancy_bias * resp["agreement"]
        + format_bias * resp["formatting"]
        + confidence_bias * resp["confidence"]
    )
    total = tq + bias_part
    labels.append(resp["label"])
    true_quals.append(tq)
    total_rewards.append(total)
    bias_contributions.append(bias_part)

# Bar chart
fig_bias = go.Figure()
fig_bias.add_trace(go.Bar(
    name="True Quality",
    x=labels,
    y=true_quals,
    marker_color=COLORS["blue"],
    opacity=0.85,
))
fig_bias.add_trace(go.Bar(
    name="Bias Contribution",
    x=labels,
    y=bias_contributions,
    marker_color=COLORS["orange"],
    opacity=0.85,
))
fig_bias.update_layout(
    barmode="stack",
    title="Total Reward = True Quality + Bias Contributions",
    yaxis_title="Reward Score",
    xaxis_tickangle=-20,
    height=420,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
)
st.plotly_chart(fig_bias, use_container_width=True)

# Ranking comparison
true_ranking = sorted(range(len(labels)), key=lambda i: true_quals[i], reverse=True)
biased_ranking = sorted(range(len(labels)), key=lambda i: total_rewards[i], reverse=True)

col_rank_t, col_rank_b = st.columns(2)
with col_rank_t:
    st.markdown("**Ranking by True Quality**")
    for rank, idx in enumerate(true_ranking):
        st.markdown(f"{rank+1}. {labels[idx]} ({true_quals[idx]:.2f})")
with col_rank_b:
    st.markdown("**Ranking by Biased Reward**")
    for rank, idx in enumerate(biased_ranking):
        color = COLORS["green"] if biased_ranking[rank] == true_ranking[rank] else COLORS["red"]
        st.markdown(
            f"{rank+1}. {labels[idx]} "
            f"({total_rewards[idx]:.2f})"
        )

if any(b > 0 for b in [length_bias, sycophancy_bias, format_bias, confidence_bias]):
    rank_distortion = sum(
        1 for i in range(len(labels)) if true_ranking[i] != biased_ranking[i]
    )
    st.markdown(f"""
<div class="insight-box">
<strong>Rank distortion:</strong> {rank_distortion}/{len(labels)} positions changed.
Biases shift the ranking away from true quality, creating exploitable reward signal.
</div>
""", unsafe_allow_html=True)


# ── 5. Goodhart's Law Demo ───────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Goodhart's Law")
st.markdown(
    '"When a measure becomes a target, it ceases to be a good measure." '
    "As we optimize harder against the reward model (increasing KL from the "
    "base policy), the proxy reward keeps climbing but true quality peaks "
    "and then collapses."
)

kl_budget = st.slider(
    "Optimization pressure (KL budget)",
    0.0, 30.0, 10.0, 0.5,
    help="Higher KL = more aggressive optimization against the reward model."
)

# Simulate proxy reward and true quality as function of KL
kl_range = np.linspace(0, 30, 300)

# Proxy reward: monotonically increasing (reward model is being optimized)
proxy_reward = 2.0 * np.tanh(0.15 * kl_range) + 0.3 * kl_range * 0.05

# True quality: rises then falls -- Goodhart's law
true_quality = 1.8 * np.tanh(0.2 * kl_range) - 0.025 * (kl_range ** 1.3)
true_quality = true_quality - true_quality[0]  # start at 0

# Normalize for visualization
proxy_reward = proxy_reward / max(abs(proxy_reward.max()), 1e-10)
true_quality_norm = true_quality / max(abs(true_quality[:100].max()), 1e-10) * proxy_reward[:100].max()

# Find where user's KL slider is
kl_idx = np.argmin(np.abs(kl_range - kl_budget))
current_proxy = proxy_reward[kl_idx]
current_true = true_quality_norm[kl_idx]

# Find the peak of true quality
peak_idx = np.argmax(true_quality_norm)
peak_kl = kl_range[peak_idx]

fig_goodhart = go.Figure()

fig_goodhart.add_trace(go.Scatter(
    x=kl_range, y=proxy_reward,
    name="Proxy Reward (RM)",
    line=dict(color=COLORS["blue"], width=3),
))
fig_goodhart.add_trace(go.Scatter(
    x=kl_range, y=true_quality_norm,
    name="True Quality",
    line=dict(color=COLORS["green"], width=3),
))

# Shaded region: overoptimization zone
fig_goodhart.add_vrect(
    x0=peak_kl, x1=30,
    fillcolor=COLORS["red"], opacity=0.08,
    layer="below", line_width=0,
    annotation_text="Overoptimization Zone",
    annotation_position="top right",
    annotation=dict(font=dict(color=COLORS["red"], size=12)),
)

# Current KL marker
fig_goodhart.add_trace(go.Scatter(
    x=[kl_budget], y=[current_proxy],
    mode="markers", marker=dict(size=14, color=COLORS["blue"], symbol="circle",
                                 line=dict(width=2, color=COLORS["white"])),
    name=f"Current Proxy: {current_proxy:.3f}",
))
fig_goodhart.add_trace(go.Scatter(
    x=[kl_budget], y=[current_true],
    mode="markers", marker=dict(size=14, color=COLORS["green"], symbol="diamond",
                                 line=dict(width=2, color=COLORS["white"])),
    name=f"Current True: {current_true:.3f}",
))

# Vertical line at peak
fig_goodhart.add_vline(
    x=peak_kl, line_dash="dot", line_color=COLORS["orange"], line_width=2,
    annotation_text=f"Optimal KL = {peak_kl:.1f}",
    annotation_position="top left",
    annotation=dict(font=dict(color=COLORS["orange"], size=11)),
)

fig_goodhart.update_layout(
    title="Goodhart's Law: Proxy Reward vs True Quality",
    xaxis_title="KL Divergence from Base Policy",
    yaxis_title="Normalized Score",
    height=480,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
)

st.plotly_chart(fig_goodhart, use_container_width=True)

gap = current_proxy - current_true
gap_color = COLORS["green"] if kl_budget <= peak_kl else COLORS["red"]
status = "Within safe optimization range" if kl_budget <= peak_kl else "In overoptimization territory"

col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">Proxy - True Gap</span><br/>
<span style="font-size:2rem;font-weight:700;color:{gap_color};">{gap:.3f}</span>
</div>
""", unsafe_allow_html=True)
with col_g2:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">Optimal KL</span><br/>
<span style="font-size:2rem;font-weight:700;color:{COLORS['orange']};">{peak_kl:.1f}</span>
</div>
""", unsafe_allow_html=True)
with col_g3:
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']};font-size:0.85rem;">Status</span><br/>
<span style="font-size:1.1rem;font-weight:600;color:{gap_color};">{status}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
<strong>Goodhart's Law in action:</strong> The proxy reward (from the RM) keeps climbing to
{proxy_reward[-1]:.2f} at maximum KL, but true quality peaks at KL = {peak_kl:.1f} and then
<em>declines</em>. Past the optimum, further optimization exploits imperfections in the RM
rather than improving actual quality. This is why KL constraints are essential in RLHF.
</div>
""", unsafe_allow_html=True)


# ── 6. Key Insight ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Key Insight")

st.markdown("""
<div class="insight-box" style="font-size:1.1rem; padding:24px 28px;">
<strong>The reward model is a lossy compressor.</strong><br/><br/>
Human preferences are rich, contextual, and multi-dimensional. The RM compresses
all of that into a single scalar <code>r(x, y)</code>. This compression is inherently
lossy:<br/><br/>
<code style="font-size:1.05rem;">&nbsp; K(true human preferences) >> K(reward model)</code><br/><br/>
The gap between the Kolmogorov complexity of true preferences and the reward model's
capacity is exactly where <strong>reward hacking</strong> lives. The policy finds outputs
that score high under the RM's simplified model of quality without actually being high
quality -- exploiting the compression artifacts.<br/><br/>
This is why post-training research keeps searching for richer reward signals: process
reward models, constitutional AI, debate, scalable oversight. Each tries to close the
gap between K(true preferences) and K(reward signal).
</div>
""", unsafe_allow_html=True)
