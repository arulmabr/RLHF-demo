"""
Page 10 -- RL on Verifiable Rewards
r(x,y) = 1 if correct, 0 if wrong.  No learned proxy.  No Goodhart's Law.
"""

from style import inject_custom_css, COLORS
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION X</p>',
    unsafe_allow_html=True,
)
st.title("RL on Verifiable Rewards")

st.markdown(
    """
    The simplest possible reward signal: **did the model get the right answer?**

    $$r(x, y) = \\begin{cases} 1 & \\text{if } y \\text{ is correct} \\\\ 0 & \\text{otherwise} \\end{cases}$$

    No learned proxy.  No reward model that can be hacked.  Just a binary
    ground-truth check.  This works for domains where correctness is
    **mechanically verifiable**: math, code execution, formal proofs, factual
    lookups with known answers.
    """
)

st.markdown(
    '<div class="big-formula">'
    "r(x, y) = 1 if correct, 0 if wrong &mdash; no learned proxy, no Goodhart's Law"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    The challenge is **extreme sparsity**.  A model generating a 10,000-token
    chain-of-thought gets a single bit of feedback at the very end: right or wrong.
    Credit assignment -- figuring out *which* reasoning steps helped or hurt --
    becomes nearly impossible from the outcome reward alone.
    """
)

st.markdown("---")

# =====================================================================
# 1. SPARSE REWARD CHALLENGE DEMO
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Sparse Reward Challenge")
st.markdown(
    "Simulate a math-solving agent working through a multi-step problem. "
    "Only the **final answer** receives a reward (0 or 1). Toggle individual "
    "steps to see how intermediate correctness is invisible to the outcome signal."
)

reasoning_steps = [
    ("Parse the problem statement", "Extract variables and constraints"),
    ("Identify relevant formulas", "Select equations to apply"),
    ("Set up equation system", "Translate words to algebra"),
    ("Simplify left-hand side", "Combine like terms"),
    ("Isolate variable x", "Move terms across equals sign"),
    ("Substitute back into eq. 2", "Reduce to single variable"),
    ("Expand the product", "Apply distributive property"),
    ("Collect constant terms", "Arithmetic on constants"),
    ("Factor the quadratic", "Find factors of leading term"),
    ("Apply quadratic formula", "Compute discriminant"),
    ("Evaluate square root", "Simplify radical expression"),
    ("Select positive root", "Apply domain constraint"),
    ("Compute y from x", "Back-substitution"),
    ("Simplify the fraction", "Reduce to lowest terms"),
    ("Verify units", "Dimensional analysis check"),
    ("Check boundary conditions", "Test edge cases"),
    ("Round to required precision", "Apply formatting rules"),
    ("Verify against original", "Plug back into original equation"),
    ("Format final answer", "Box the answer"),
    ("Submit answer", "Final output"),
]

st.markdown("**Toggle individual reasoning steps as correct or incorrect:**")

num_steps = len(reasoning_steps)
step_correct = []

# Display steps in two columns for compactness
col_left, col_right = st.columns(2)
half = num_steps // 2

for i, (step_name, step_desc) in enumerate(reasoning_steps):
    target_col = col_left if i < half else col_right
    with target_col:
        val = st.checkbox(
            f"Step {i + 1}: {step_name}",
            value=True,
            key=f"step_{i}",
            help=step_desc,
        )
        step_correct.append(val)

# Final reward is 1 only if the very last step (submit answer) is correct
# In reality, the reward only sees the final answer, not intermediate steps
final_answer_correct = step_correct[-1]
final_reward = 1 if final_answer_correct else 0

num_correct_steps = sum(step_correct)
num_wrong_steps = num_steps - num_correct_steps

st.markdown("---")

met1, met2, met3 = st.columns(3)
met1.metric("Correct Steps", f"{num_correct_steps} / {num_steps}")
met2.metric("Wrong Steps", f"{num_wrong_steps}")
met3.metric(
    "Outcome Reward",
    f"{final_reward}",
    delta="Correct" if final_reward == 1 else "Wrong",
    delta_color="normal" if final_reward == 1 else "inverse",
)

# Visualization: step-by-step correctness vs what the reward sees
step_labels = [f"S{i + 1}" for i in range(num_steps)]
step_colors = [
    COLORS["green"] if c else COLORS["red"] for c in step_correct
]

# What an outcome-only reward can infer about each step
# The reward only knows the final result -- it gives uniform "credit" to all steps
uniform_credit = [final_reward / num_steps] * num_steps

fig_sparse = go.Figure()

# True step correctness (invisible to reward)
fig_sparse.add_trace(go.Bar(
    x=step_labels,
    y=[1 if c else 0 for c in step_correct],
    name="True Step Correctness (hidden from reward)",
    marker_color=step_colors,
    opacity=0.85,
))

# What the outcome reward can assign
fig_sparse.add_trace(go.Scatter(
    x=step_labels,
    y=uniform_credit,
    mode="lines+markers",
    name=f"Outcome Reward Credit (= {final_reward}/{num_steps} per step)",
    line=dict(color=COLORS["orange"], width=3, dash="dash"),
    marker=dict(size=7, color=COLORS["orange"]),
))

fig_sparse.update_layout(
    title="Credit Assignment Problem: True Correctness vs Outcome Reward",
    xaxis_title="Reasoning Step",
    yaxis_title="Value",
    yaxis=dict(range=[-0.05, 1.15]),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_sparse, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The credit assignment problem:</strong> The outcome reward is a "
    "single binary signal for the entire chain. Even if 19 out of 20 steps were "
    "correct, one wrong final answer yields reward 0 -- and the model cannot "
    "tell which step caused the failure. Conversely, a lucky final answer gives "
    "reward 1 to all steps, even the flawed ones. This is the fundamental "
    "challenge of training on verifiable rewards."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 2. EMERGENCE THRESHOLD VISUALIZATION
# =====================================================================
st.markdown(
    '<p class="section-header">VISUALIZATION</p>',
    unsafe_allow_html=True,
)
st.subheader("Emergence Threshold")
st.markdown(
    "With verifiable rewards, accuracy on a problem class stays near **zero** "
    "until the model has enough reasoning capability to solve it -- then it "
    "**jumps sharply**. Easy problems emerge first, hard problems require "
    "much more compute."
)

compute = st.slider(
    "Training compute (arbitrary units)",
    min_value=0,
    max_value=100,
    value=40,
    step=1,
    key="compute_slider",
)

# Emergence curves: sigmoid with different thresholds
compute_axis = np.linspace(0, 100, 500)


def emergence_curve(x, threshold, steepness=0.25):
    """Sigmoid emergence: near-zero below threshold, rapid jump at threshold."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))


difficulties = {
    "Easy (arithmetic, basic algebra)": {"threshold": 20, "steepness": 0.35, "color": COLORS["green"]},
    "Medium (word problems, proofs)": {"threshold": 50, "steepness": 0.25, "color": COLORS["orange"]},
    "Hard (competition math, research)": {"threshold": 80, "steepness": 0.20, "color": COLORS["red"]},
}

fig_emerge = go.Figure()

for label, params in difficulties.items():
    y_vals = emergence_curve(compute_axis, params["threshold"], params["steepness"])
    fig_emerge.add_trace(go.Scatter(
        x=compute_axis,
        y=y_vals * 100,
        mode="lines",
        name=label,
        line=dict(color=params["color"], width=3),
    ))

# Vertical line at current compute
fig_emerge.add_vline(
    x=compute,
    line_width=2,
    line_dash="dash",
    line_color=COLORS["cyan"],
    annotation_text=f"Compute = {compute}",
    annotation_position="top right",
    annotation_font_color=COLORS["cyan"],
)

fig_emerge.update_layout(
    title="Accuracy vs Training Compute by Problem Difficulty",
    xaxis_title="Training Compute",
    yaxis_title="Accuracy (%)",
    yaxis=dict(range=[-2, 108]),
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_emerge, use_container_width=True)

# Metrics at current compute
easy_acc = emergence_curve(compute, 20, 0.35) * 100
med_acc = emergence_curve(compute, 50, 0.25) * 100
hard_acc = emergence_curve(compute, 80, 0.20) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Easy Accuracy", f"{easy_acc:.1f}%")
c2.metric("Medium Accuracy", f"{med_acc:.1f}%")
c3.metric("Hard Accuracy", f"{hard_acc:.1f}%")

st.markdown(
    '<div class="insight-box">'
    "<strong>Why emergence?</strong> With binary rewards, partial credit does not "
    "exist. A model that gets 80% of the reasoning right but stumbles at the end "
    "receives the same reward (0) as one that gets nothing right. Accuracy stays "
    "near zero until the model can reliably complete the <em>entire</em> reasoning "
    "chain -- then it jumps. This is the signature of verifiable rewards: "
    "capability appears to emerge suddenly."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. VERIFIABLE VS PROXY REWARD COMPARISON
# =====================================================================
st.markdown(
    '<p class="section-header">COMPARISON</p>',
    unsafe_allow_html=True,
)
st.subheader("Verifiable vs Proxy Reward")
st.markdown(
    "Two fundamentally different reward regimes. **Verifiable rewards** cannot "
    "be hacked -- if the answer is wrong, the reward is 0, period. **Proxy "
    "rewards** from a learned reward model can diverge from true quality."
)

training_steps = np.linspace(0, 1000, 500)

# Verifiable reward regime
verif_reward = 1.0 - np.exp(-0.004 * training_steps)  # saturates at 1.0
verif_quality = verif_reward.copy()  # reward IS quality -- they are identical

# Proxy reward regime
proxy_reward = 0.6 * (1 - np.exp(-0.005 * training_steps)) + 0.0008 * training_steps
proxy_true_quality = (
    0.8 * (1 - np.exp(-0.005 * training_steps))
    - 0.0000008 * training_steps ** 2
)
proxy_true_quality = np.maximum(proxy_true_quality, 0)

col_verif, col_proxy = st.columns(2)

with col_verif:
    st.markdown(
        '<div class="concept-card">'
        f'<strong style="color:{COLORS["green"]};">Verifiable Rewards</strong><br/>'
        "Reward = ground truth correctness.<br/>"
        "Cannot be hacked or gamed.<br/>"
        "Quality improves monotonically.<br/>"
        "Limited to verifiable domains."
        "</div>",
        unsafe_allow_html=True,
    )

    fig_verif = go.Figure()
    fig_verif.add_trace(go.Scatter(
        x=training_steps, y=verif_reward,
        mode="lines", name="Reward (= quality)",
        line=dict(color=COLORS["green"], width=3),
    ))
    fig_verif.add_trace(go.Scatter(
        x=training_steps, y=verif_quality,
        mode="lines", name="True Quality",
        line=dict(color=COLORS["blue"], width=2, dash="dot"),
        visible="legendonly",
    ))
    fig_verif.update_layout(
        title="Verifiable: Reward = Quality",
        xaxis_title="Training Steps",
        yaxis_title="Score",
        yaxis=dict(range=[-0.05, 1.35]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Annotation: they are the same line
    fig_verif.add_annotation(
        x=700, y=verif_reward[int(700 / 1000 * 499)] + 0.08,
        text="Reward and quality<br>are identical",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["green"],
        font=dict(color=COLORS["green"], size=12),
        ax=0, ay=-40,
    )
    st.plotly_chart(fig_verif, use_container_width=True)

with col_proxy:
    st.markdown(
        '<div class="concept-card">'
        f'<strong style="color:{COLORS["red"]};">Proxy Rewards (Learned RM)</strong><br/>'
        "Reward = learned approximation.<br/>"
        "Susceptible to Goodhart's Law.<br/>"
        "Quality diverges from reward over time.<br/>"
        "Applies to any domain."
        "</div>",
        unsafe_allow_html=True,
    )

    fig_proxy = go.Figure()
    fig_proxy.add_trace(go.Scatter(
        x=training_steps, y=proxy_reward,
        mode="lines", name="Proxy Reward (RM)",
        line=dict(color=COLORS["red"], width=3),
    ))
    fig_proxy.add_trace(go.Scatter(
        x=training_steps, y=proxy_true_quality,
        mode="lines", name="True Quality",
        line=dict(color=COLORS["blue"], width=3, dash="dash"),
    ))
    fig_proxy.update_layout(
        title="Proxy: Reward Diverges from Quality",
        xaxis_title="Training Steps",
        yaxis_title="Score",
        yaxis=dict(range=[-0.05, 1.35]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Mark the divergence point
    divergence_idx = np.argmax(proxy_true_quality)
    fig_proxy.add_annotation(
        x=training_steps[divergence_idx],
        y=proxy_true_quality[divergence_idx] + 0.08,
        text="Quality peaks<br>then degrades",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["orange"],
        font=dict(color=COLORS["orange"], size=12),
        ax=50, ay=-40,
    )
    fig_proxy.add_annotation(
        x=900,
        y=proxy_reward[int(900 / 1000 * 499)] + 0.05,
        text="Reward keeps<br>climbing (hacked)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["red"],
        font=dict(color=COLORS["red"], size=11),
        ax=0, ay=-40,
    )
    st.plotly_chart(fig_proxy, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The fundamental difference:</strong> With verifiable rewards, "
    "the reward signal <em>is</em> the ground truth. There is no gap between "
    "what the reward measures and what we actually want. With proxy rewards, "
    "the learned reward model is an imperfect approximation -- and the policy "
    "will eventually find and exploit the gap."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 4. DOMAIN APPLICABILITY
# =====================================================================
st.markdown(
    '<p class="section-header">DOMAIN ANALYSIS</p>',
    unsafe_allow_html=True,
)
st.subheader("Domain Applicability")
st.markdown(
    "Not all tasks have verifiable rewards. The method's effectiveness "
    "depends on how objectively you can check the answer."
)

domains = [
    ("Math (computation)", 0.98, True),
    ("Code (unit tests)", 0.95, True),
    ("Formal logic / proofs", 0.92, True),
    ("Factual Q&A (known)", 0.85, True),
    ("Translation", 0.55, False),
    ("Summarization", 0.40, False),
    ("Creative writing", 0.15, False),
    ("Empathy / emotional support", 0.10, False),
    ("Nuanced conversation", 0.08, False),
    ("Humor / wit", 0.12, False),
]

domain_names = [d[0] for d in domains]
domain_scores = [d[1] for d in domains]
domain_verifiable = [d[2] for d in domains]

domain_colors = [
    COLORS["green"] if v else COLORS["red"] for v in domain_verifiable
]

fig_domains = go.Figure()
fig_domains.add_trace(go.Bar(
    y=domain_names,
    x=domain_scores,
    orientation="h",
    marker_color=domain_colors,
    text=[f"{s:.0%}" for s in domain_scores],
    textposition="outside",
))

fig_domains.update_layout(
    title="Reward Verifiability by Domain",
    xaxis_title="Verifiability Score",
    xaxis=dict(range=[0, 1.15], tickformat=".0%"),
    yaxis=dict(autorange="reversed"),
    height=460,
    showlegend=False,
    margin=dict(l=200, r=50, t=50, b=40),
)

# Add a dividing annotation
fig_domains.add_shape(
    type="line",
    x0=0.70, x1=0.70,
    y0=-0.5, y1=len(domains) - 0.5,
    line=dict(color=COLORS["orange"], width=2, dash="dash"),
)
fig_domains.add_annotation(
    x=0.70, y=-0.8,
    text="Verifiability threshold",
    font=dict(color=COLORS["orange"], size=11),
    showarrow=False,
    yshift=10,
)

st.plotly_chart(fig_domains, use_container_width=True)

col_dom1, col_dom2 = st.columns(2)
with col_dom1:
    st.markdown(
        '<div class="concept-card">'
        f'<strong style="color:{COLORS["green"]};">Verifiable Domains</strong>'
        "<ul>"
        + "".join(f"<li>{d[0]} ({d[1]:.0%})</li>" for d in domains if d[2])
        + "</ul>"
        "Binary correctness check is possible. "
        "RL on verifiable rewards works directly."
        "</div>",
        unsafe_allow_html=True,
    )
with col_dom2:
    st.markdown(
        '<div class="concept-card">'
        f'<strong style="color:{COLORS["red"]};">Non-Verifiable Domains</strong>'
        "<ul>"
        + "".join(f"<li>{d[0]} ({d[1]:.0%})</li>" for d in domains if not d[2])
        + "</ul>"
        "Quality is subjective. "
        "Requires learned reward models (with Goodhart risk)."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# Slider: domain verifiability and method effectiveness
st.markdown("##### Method Effectiveness vs Domain Verifiability")

verifiability = st.slider(
    "Domain verifiability",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    format="%.0f%%",
    key="verifiability_slider",
    help="0% = purely subjective, 100% = fully verifiable",
)

verif_axis = np.linspace(0, 1, 200)

# Verifiable RL effectiveness: high when verifiable, drops off sharply
verif_rl_eff = np.where(
    verif_axis > 0.6,
    0.3 + 0.7 * ((verif_axis - 0.6) / 0.4) ** 0.8,
    0.3 * (verif_axis / 0.6) ** 2,
)

# RLHF effectiveness: works reasonably across the board but plateaus
rlhf_eff = 0.55 * (1 - np.exp(-3.0 * verif_axis)) + 0.15

# Combined / best approach
best_eff = np.maximum(verif_rl_eff, rlhf_eff)

fig_eff = go.Figure()

fig_eff.add_trace(go.Scatter(
    x=verif_axis * 100, y=verif_rl_eff * 100,
    mode="lines", name="RL on Verifiable Rewards",
    line=dict(color=COLORS["green"], width=3),
))
fig_eff.add_trace(go.Scatter(
    x=verif_axis * 100, y=rlhf_eff * 100,
    mode="lines", name="RLHF (Proxy Reward)",
    line=dict(color=COLORS["red"], width=3, dash="dash"),
))
fig_eff.add_trace(go.Scatter(
    x=verif_axis * 100, y=best_eff * 100,
    mode="lines", name="Best Available Method",
    line=dict(color=COLORS["cyan"], width=2, dash="dot"),
))

fig_eff.add_vline(
    x=verifiability * 100,
    line_width=2,
    line_dash="dash",
    line_color=COLORS["orange"],
    annotation_text=f"Verifiability = {verifiability:.0%}",
    annotation_position="top right",
    annotation_font_color=COLORS["orange"],
)

fig_eff.update_layout(
    title="Method Effectiveness vs Domain Verifiability",
    xaxis_title="Domain Verifiability (%)",
    yaxis_title="Training Effectiveness (%)",
    yaxis=dict(range=[-2, 108]),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_eff, use_container_width=True)

# Metrics at current verifiability
v_idx = int(verifiability * 199)
e1, e2 = st.columns(2)
e1.metric("Verifiable RL Effectiveness", f"{verif_rl_eff[v_idx] * 100:.1f}%")
e2.metric("RLHF Effectiveness", f"{rlhf_eff[v_idx] * 100:.1f}%")

st.markdown(
    '<div class="insight-box">'
    "<strong>The crossover:</strong> For highly verifiable domains (math, code), "
    "RL on verifiable rewards dominates -- it is immune to reward hacking and "
    "scales cleanly. For subjective domains (creative writing, empathy), RLHF "
    "with a learned proxy is the only option. The frontier of research is pushing "
    "the verifiability boundary further: can we make more tasks verifiable?"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 5. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.1rem; padding:24px 28px;">'
    "<strong>Key Insight</strong><br/><br/>"
    "No learned proxy. No Goodhart's Law. The reward is ground truth: "
    "1 if the answer is correct, 0 if it is not.<br/><br/>"
    "The challenge is <strong>extreme sparsity</strong> &mdash; a single binary "
    "signal at the end of thousands of tokens of reasoning. But when it works, "
    "there is <em>nothing to hack</em>. The policy improves monotonically toward "
    "genuine capability, and the only limit is how much compute you can afford."
    "</div>",
    unsafe_allow_html=True,
)
