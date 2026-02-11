"""
Page 7 -- DPO: Direct Preference Optimization
The policy IS the reward model.
"""

from style import inject_custom_css, COLORS, sigmoid, softmax, kl_divergence
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION VII</p>',
    unsafe_allow_html=True,
)
st.title("DPO: Direct Preference Optimization")

st.markdown(
    """
    RLHF works, but it is a Rube Goldberg machine: train a reward model, spin
    up a PPO loop with clipping and value heads, tune dozens of hyperparameters,
    and pray for stability. **DPO asks: what if we could skip all of that?**

    The key mathematical insight is that the optimal policy under a KL-constrained
    reward-maximization objective has a closed-form relationship with the reward:
    """
)

st.markdown(
    '<div class="big-formula">'
    "r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    Substitute this into the Bradley-Terry preference model
    P(y_w > y_l) = sigma(r(y_w) - r(y_l)). The partition function Z(x) cancels,
    giving us a loss that depends **only on the policy and the reference**, with
    no reward model anywhere:
    """
)

st.markdown(
    '<div class="big-formula">'
    "L_DPO = -E[ log sigma( beta * ( log pi_theta(y_w|x)/pi_ref(y_w|x)"
    " - log pi_theta(y_l|x)/pi_ref(y_l|x) ) ) ]"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    No reward model. No RL loop. No value function. No GAE. No clipping.
    Just supervised learning on preference pairs.

    The implicit reward under DPO is:
    """
)

st.markdown(
    '<div class="big-formula">'
    "r_DPO(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="insight-box">'
    "<strong>The policy IS the reward model.</strong> Every time DPO updates "
    "the policy weights, it implicitly reshapes the reward landscape. There is "
    "no separate reward function to train, store, or query -- the log-probability "
    "ratio between the trained policy and the reference policy <em>is</em> the reward."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 1. IMPLICIT REWARD VISUALIZER
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Implicit Reward Visualizer")
st.markdown(
    "The implicit reward for any response is simply "
    "**beta * log(pi_theta / pi_ref)**. Below we simulate several response "
    "types with different log-probability ratios. Watch how beta scales the "
    "reward signal and how DPO simultaneously pushes up preferred and pushes "
    "down dispreferred responses."
)

beta_reward = st.slider(
    "beta (KL penalty strength)",
    min_value=0.05,
    max_value=1.0,
    value=0.1,
    step=0.01,
    format="%.2f",
    key="beta_reward",
)

# Simulated response types: (label, log_pi_theta, log_pi_ref)
response_types = [
    ("Preferred (chosen)", -1.2, -2.0),
    ("Slightly preferred", -1.5, -1.8),
    ("Neutral", -2.0, -2.0),
    ("Slightly dispreferred", -2.5, -2.0),
    ("Dispreferred (rejected)", -3.5, -2.0),
]

labels = [r[0] for r in response_types]
log_ratios = [r[1] - r[2] for r in response_types]
implicit_rewards = [beta_reward * lr for lr in log_ratios]

bar_colors = [
    COLORS["green"] if r > 0.01
    else (COLORS["red"] if r < -0.01 else COLORS["gray"])
    for r in implicit_rewards
]

fig_reward = go.Figure()

fig_reward.add_trace(go.Bar(
    y=labels,
    x=implicit_rewards,
    orientation="h",
    marker_color=bar_colors,
    text=[f"{r:+.3f}" for r in implicit_rewards],
    textposition="outside",
    name="Implicit Reward",
))

# Add a vertical zero line
fig_reward.add_vline(x=0, line_dash="dash", line_color=COLORS["gray"], line_width=1)

fig_reward.update_layout(
    title=f"Implicit Reward r(x,y) = {beta_reward:.2f} * log(pi_theta / pi_ref)",
    xaxis_title="Implicit Reward",
    yaxis=dict(autorange="reversed"),
    height=380,
    showlegend=False,
    margin=dict(l=180, r=60, t=50, b=40),
)
st.plotly_chart(fig_reward, use_container_width=True)

# Show the log-ratio detail table
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#2ECC71;">Preferred responses</strong><br>'
        "Have <code>log(pi_theta/pi_ref) > 0</code>: the policy assigns them "
        "<em>more</em> probability than the reference did. "
        "DPO pushes these up further."
        "</div>",
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#E74C3C;">Dispreferred responses</strong><br>'
        "Have <code>log(pi_theta/pi_ref) < 0</code>: the policy assigns them "
        "<em>less</em> probability than the reference. "
        "DPO pushes these down further."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 2. DPO GRADIENT WEIGHT DEMO
# =====================================================================
st.markdown(
    '<p class="section-header">GRADIENT ANALYSIS</p>',
    unsafe_allow_html=True,
)
st.subheader("DPO Gradient Weight: Adaptive Curriculum")
st.markdown(
    """
    The DPO gradient takes the form:

    > nabla L ~ **-w** * [ nabla log pi_theta(y_w) - nabla log pi_theta(y_l) ]

    where the weight **w** depends on how much the model currently agrees with the
    preference label. Define **u** = beta * (log pi_theta(y_w)/pi_ref(y_w) -
    log pi_theta(y_l)/pi_ref(y_l)) as the model's current implicit reward margin.
    Then:
    """
)

st.markdown(
    '<div class="big-formula">'
    "w = sigma(-u) = 1 / (1 + exp(u))"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "When **u is large** (model already agrees with the preference), "
    "sigma(-u) is small and the gradient vanishes -- no need to fix what "
    "already works. When **u is small or negative** (model disagrees), "
    "the weight is large and the gradient pushes hard. This creates a "
    "natural **curriculum**: DPO focuses on the examples it gets wrong."
)

beta_grad = st.slider(
    "beta for gradient visualization",
    min_value=0.05,
    max_value=1.0,
    value=0.2,
    step=0.01,
    key="beta_grad",
)

u_vals = np.linspace(-6, 6, 300)
weights = sigmoid(-u_vals)

fig_grad = go.Figure()

fig_grad.add_trace(go.Scatter(
    x=u_vals,
    y=weights,
    mode="lines",
    line=dict(color=COLORS["cyan"], width=3),
    name="Gradient weight w = sigma(-u)",
    hovertemplate="u = %{x:.2f}<br>weight = %{y:.3f}<extra></extra>",
))

# Annotate regions
fig_grad.add_vrect(
    x0=-6, x1=-1,
    fillcolor=COLORS["red"], opacity=0.08,
    layer="below", line_width=0,
)
fig_grad.add_vrect(
    x0=1, x1=6,
    fillcolor=COLORS["green"], opacity=0.08,
    layer="below", line_width=0,
)

fig_grad.add_annotation(
    x=-3.5, y=0.92, text="Model disagrees<br>(large gradient)",
    showarrow=False, font=dict(color=COLORS["red"], size=12),
)
fig_grad.add_annotation(
    x=3.5, y=0.15, text="Model agrees<br>(small gradient)",
    showarrow=False, font=dict(color=COLORS["green"], size=12),
)
fig_grad.add_annotation(
    x=0, y=0.55, text="Uncertain<br>(moderate gradient)",
    showarrow=False, font=dict(color=COLORS["orange"], size=12),
)

fig_grad.update_layout(
    title="DPO Gradient Weight vs Implicit Reward Margin",
    xaxis_title="u = beta * (log-ratio_w - log-ratio_l) [implicit margin]",
    yaxis_title="Gradient weight sigma(-u)",
    height=420,
    yaxis=dict(range=[-0.02, 1.05]),
)
st.plotly_chart(fig_grad, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Curriculum learning for free:</strong> Unlike PPO which treats "
    "every sample roughly equally (modulo advantages), DPO's gradient "
    "naturally down-weights examples the model has already learned and "
    "concentrates on the hard cases. This is why DPO often converges faster "
    "than PPO in practice."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. BETA SENSITIVITY EXPLORER
# =====================================================================
st.markdown(
    '<p class="section-header">BETA EXPLORER</p>',
    unsafe_allow_html=True,
)
st.subheader("Beta Sensitivity: The KL Thermostat")
st.markdown(
    "Beta controls the tradeoff between matching preferences and staying "
    "close to the reference policy. It is the single most important "
    "hyperparameter in DPO."
)

beta_explore = st.slider(
    "beta",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.01,
    key="beta_explore",
)

# Show regime classification
if beta_explore < 0.1:
    regime_text = (
        '<div class="concept-card" style="border-left: 4px solid #E74C3C;">'
        '<strong style="color:#E74C3C;">Danger zone: beta < 0.1</strong><br>'
        "The KL penalty is too weak. The policy can drift far from the reference, "
        "leading to <strong>mode collapse</strong>: the model may push probability "
        "mass away from <em>both</em> y_w and y_l in favor of unseen responses. "
        "Outputs become degenerate."
        "</div>"
    )
elif beta_explore > 0.5:
    regime_text = (
        '<div class="concept-card" style="border-left: 4px solid #F39C12;">'
        '<strong style="color:#F39C12;">Conservative zone: beta > 0.5</strong><br>'
        "The KL penalty dominates. The policy barely moves from the reference. "
        "Training is stable but <strong>alignment is weak</strong> -- the model "
        "has not learned much from the preferences."
        "</div>"
    )
else:
    regime_text = (
        '<div class="concept-card" style="border-left: 4px solid #2ECC71;">'
        '<strong style="color:#2ECC71;">Sweet spot: 0.1 <= beta <= 0.5</strong><br>'
        "The policy moves meaningfully from the reference while staying anchored. "
        "Preferred responses get upweighted, dispreferred get downweighted, and "
        "the model does not degenerate."
        "</div>"
    )
st.markdown(regime_text, unsafe_allow_html=True)

# DPO loss surface as a function of log-ratio difference
delta = np.linspace(-6, 6, 400)  # log_ratio_w - log_ratio_l

# Plot for several beta values including the user-selected one
beta_set = sorted(set([0.05, 0.1, 0.2, 0.5, 1.0, beta_explore]))
beta_colors = {
    0.05: COLORS["red"],
    0.1: COLORS["orange"],
    0.2: COLORS["green"],
    0.5: COLORS["blue"],
    1.0: COLORS["purple"],
}

fig_beta = go.Figure()

for b in beta_set:
    loss = -np.log(sigmoid(b * delta))
    is_selected = abs(b - beta_explore) < 1e-6
    color = beta_colors.get(b, COLORS["cyan"])
    if is_selected and b not in beta_colors:
        color = COLORS["cyan"]

    fig_beta.add_trace(go.Scatter(
        x=delta,
        y=loss,
        mode="lines",
        name=f"beta={b:.2f}" + (" (selected)" if is_selected else ""),
        line=dict(
            color=color,
            width=4 if is_selected else 2,
            dash="solid" if is_selected else "dot",
        ),
        opacity=1.0 if is_selected else 0.6,
        hovertemplate=f"beta={b:.2f}<br>delta=%{{x:.2f}}<br>loss=%{{y:.3f}}<extra></extra>",
    ))

fig_beta.add_vline(x=0, line_dash="dash", line_color=COLORS["gray"], line_width=1)

fig_beta.update_layout(
    title="DPO Loss: -log sigma(beta * delta) for Various beta",
    xaxis_title="delta = log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l))",
    yaxis_title="DPO Loss",
    height=450,
    yaxis=dict(range=[0, 5]),
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_beta, use_container_width=True)

st.markdown(
    "**Reading the plot:** The x-axis is the model's current implicit reward "
    "margin (delta). When delta > 0, the model already prefers y_w over y_l "
    "(correct). When delta < 0, the model gets it wrong. Higher beta means "
    "a steeper loss curve -- stronger preference signal, but also stronger "
    "resistance to moving away from the reference."
)

st.markdown("---")

# =====================================================================
# 4. DPO FAILURE MODES
# =====================================================================
st.markdown(
    '<p class="section-header">FAILURE MODES</p>',
    unsafe_allow_html=True,
)
st.subheader("DPO Failure Modes")
st.markdown(
    "DPO is elegant but not bulletproof. Understanding where it breaks "
    "is essential for practitioners."
)

failure_modes = [
    {
        "title": "1. The Offline Distribution Problem",
        "color": COLORS["red"],
        "icon": "stale",
        "description": (
            "DPO trains on <strong>fixed preference pairs</strong> collected "
            "from the reference policy. As training progresses, the learned "
            "policy drifts away from the reference -- but the training data "
            "does not change. The pairs become increasingly off-policy, "
            "leading to <strong>distributional mismatch</strong>. "
            "The model optimizes for responses it would never actually generate."
        ),
        "fix": "Online DPO / Iterative DPO: periodically re-sample from the "
               "current policy and re-collect preferences.",
    },
    {
        "title": "2. Mode Collapse",
        "color": COLORS["orange"],
        "icon": "collapse",
        "description": (
            "When beta is too low, the model can satisfy the DPO objective by "
            "making <em>both</em> y_w and y_l unlikely, while shifting probability "
            "mass to unseen responses. The loss only cares about the "
            "<strong>difference</strong> in log-ratios, not their absolute values. "
            "So pi_theta(y_w) = 0.001 and pi_theta(y_l) = 0.0001 is a valid "
            "(but degenerate) solution."
        ),
        "fix": "Use adequate beta, add an NLL anchor loss on y_w, or use "
               "sequence-level length normalization.",
    },
    {
        "title": "3. Beta Sensitivity",
        "color": COLORS["yellow"],
        "icon": "tuning",
        "description": (
            "Unlike PPO which has many hyperparameters but is somewhat robust "
            "to each, DPO's quality depends <strong>critically</strong> on beta. "
            "Too low causes collapse, too high prevents learning. The optimal "
            "value varies across tasks, datasets, and model sizes, and there is "
            "no reliable way to set it a priori."
        ),
        "fix": "Grid search over beta in [0.05, 0.5]. Some variants (IPO) "
               "are less sensitive.",
    },
    {
        "title": "4. Quality Gap Problem",
        "color": COLORS["purple"],
        "icon": "gap",
        "description": (
            "DPO works best when the chosen and rejected responses are "
            "<strong>close in quality</strong>. If y_w is perfect and y_l is "
            "gibberish, the signal is trivial -- the model already knows. "
            "Conversely, if both are poor, there is little to learn. The "
            "richest gradient signal comes from <strong>contrastive pairs</strong> "
            "where the margin is small but meaningful."
        ),
        "fix": "Careful data curation. Rank responses and pair adjacent ranks. "
               "Use best-of-N sampling for higher quality pairs.",
    },
]

for fm in failure_modes:
    st.markdown(
        f'<div class="concept-card" style="border-left: 4px solid {fm["color"]};">'
        f'<strong style="color:{fm["color"]};">{fm["title"]}</strong><br><br>'
        f'{fm["description"]}'
        f'<br><br><strong>Mitigation:</strong> {fm["fix"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

# Visualize the offline distribution drift
st.markdown("#### Offline Drift Visualization")
st.markdown(
    "As training progresses, the policy drifts from the reference. The "
    "training data (generated by the reference) becomes increasingly stale."
)

drift_steps = np.arange(0, 101)
ref_mean = 0.0
policy_means = ref_mean + 0.03 * drift_steps  # policy drifts over time
kl_values = 0.5 * (0.03 * drift_steps) ** 2   # KL ~ 0.5 * (mu_diff)^2 for Gaussians

fig_drift = go.Figure()

fig_drift.add_trace(go.Scatter(
    x=drift_steps,
    y=kl_values,
    mode="lines",
    line=dict(color=COLORS["red"], width=3),
    name="KL(pi_theta || pi_ref)",
    hovertemplate="Step %{x}<br>KL = %{y:.3f}<extra></extra>",
))

fig_drift.add_trace(go.Scatter(
    x=drift_steps,
    y=policy_means,
    mode="lines",
    line=dict(color=COLORS["blue"], width=2, dash="dash"),
    name="Policy drift (mean shift)",
    yaxis="y2",
    hovertemplate="Step %{x}<br>Drift = %{y:.2f}<extra></extra>",
))

fig_drift.add_hrect(
    y0=0.5, y1=kl_values.max() + 0.1,
    fillcolor=COLORS["red"], opacity=0.06,
    layer="below", line_width=0,
)
fig_drift.add_annotation(
    x=80, y=kl_values[80] + 0.15,
    text="Training data<br>increasingly stale",
    showarrow=True, arrowhead=2,
    font=dict(color=COLORS["red"], size=11),
    arrowcolor=COLORS["red"],
)

fig_drift.update_layout(
    title="Offline DPO: Policy Drift Over Training",
    xaxis_title="Training Step",
    yaxis=dict(title="KL Divergence", side="left"),
    yaxis2=dict(
        title="Policy Mean Shift",
        side="right",
        overlaying="y",
        showgrid=False,
    ),
    height=380,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_drift, use_container_width=True)

st.markdown("---")

# =====================================================================
# 5. DPO FAMILY TABLE
# =====================================================================
st.markdown(
    '<p class="section-header">THE DPO FAMILY</p>',
    unsafe_allow_html=True,
)
st.subheader("DPO Variants and Extensions")
st.markdown(
    "DPO spawned a family of algorithms, each addressing a specific limitation. "
    "The core idea -- bypass the reward model -- remains, but the loss function "
    "and training procedure evolve."
)

family_data = [
    {
        "name": "DPO",
        "year": "2023",
        "key_change": "Original. Uses Bradley-Terry + closed-form reward substitution.",
        "loss_idea": "-log sigma(beta * delta)",
        "advantage": "Simple, stable, no RM needed",
        "limitation": "Offline, beta-sensitive",
    },
    {
        "name": "IPO",
        "year": "2023",
        "key_change": "Replaces log-sigmoid with a squared hinge loss. Avoids overfitting to deterministic preferences.",
        "loss_idea": "(delta - 1/(2*beta))^2",
        "advantage": "Less beta-sensitive, handles noisy labels",
        "limitation": "Slightly weaker on clean data",
    },
    {
        "name": "KTO",
        "year": "2024",
        "key_change": "Drops the pairwise requirement. Works with binary (good/bad) labels using Kahneman-Tversky value functions.",
        "loss_idea": "Separate losses for desirable vs undesirable",
        "advantage": "No paired data needed, 2x data efficiency",
        "limitation": "Weaker signal than pairwise comparison",
    },
    {
        "name": "ORPO",
        "year": "2024",
        "key_change": "Merges SFT and preference optimization into a single objective. Adds an odds-ratio penalty to the NLL loss.",
        "loss_idea": "NLL + lambda * log-odds penalty",
        "advantage": "No separate SFT stage, simpler pipeline",
        "limitation": "Harder to tune the lambda tradeoff",
    },
    {
        "name": "SimPO",
        "year": "2024",
        "key_change": "Uses average log-probability (length-normalized) as implicit reward. No reference model needed.",
        "loss_idea": "-log sigma(beta/|y| * (log pi(y_w) - log pi(y_l)) - gamma)",
        "advantage": "No reference model, length-robust",
        "limitation": "The gamma margin requires tuning",
    },
    {
        "name": "Online DPO",
        "year": "2024",
        "key_change": "Periodically re-samples from the current policy and collects fresh preferences. Fixes the offline distribution problem.",
        "loss_idea": "Same as DPO but on on-policy data",
        "advantage": "Fixes distribution drift, stronger alignment",
        "limitation": "Requires live generation + annotation",
    },
]

# Render as styled HTML table
header_row = (
    "<tr>"
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Method</th>'
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Year</th>'
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Key Change</th>'
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Loss Idea</th>'
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Advantage</th>'
    '<th style="padding:10px 14px; color:#F39C12; text-align:left;">Limitation</th>'
    "</tr>"
)

body_rows = ""
for i, entry in enumerate(family_data):
    bg = "#1E2130" if i % 2 == 0 else "#252840"
    body_rows += (
        f'<tr style="background:{bg};">'
        f'<td style="padding:10px 14px; font-weight:600; color:#4A90D9;">{entry["name"]}</td>'
        f'<td style="padding:10px 14px;">{entry["year"]}</td>'
        f'<td style="padding:10px 14px;">{entry["key_change"]}</td>'
        f'<td style="padding:10px 14px; font-family:monospace; font-size:0.85rem;">{entry["loss_idea"]}</td>'
        f'<td style="padding:10px 14px; color:#2ECC71;">{entry["advantage"]}</td>'
        f'<td style="padding:10px 14px; color:#E74C3C;">{entry["limitation"]}</td>'
        f"</tr>"
    )

table_html = (
    '<div style="overflow-x:auto; margin:12px 0;">'
    '<table style="width:100%; border-collapse:collapse; border-radius:10px; '
    'overflow:hidden; font-size:0.9rem; color:#ECF0F1;">'
    f"<thead style='background:#161829;'>{header_row}</thead>"
    f"<tbody>{body_rows}</tbody>"
    "</table></div>"
)

st.markdown(table_html, unsafe_allow_html=True)

# Visual comparison: DPO family as a scatter plot
fig_family = go.Figure()

# Axes: x = complexity of pipeline, y = data requirement
methods = [
    ("DPO",        1.0, 3.0, "Offline pairs"),
    ("IPO",        1.0, 3.0, "Offline pairs, robust loss"),
    ("KTO",        0.8, 1.5, "Unpaired binary labels"),
    ("ORPO",       0.5, 3.0, "Merged SFT + pref"),
    ("SimPO",      0.7, 3.0, "No reference model"),
    ("Online DPO", 2.5, 4.0, "On-policy pairs"),
    ("RLHF/PPO",   4.0, 4.5, "Full RL pipeline"),
]

method_colors = [
    COLORS["blue"], COLORS["cyan"], COLORS["green"],
    COLORS["orange"], COLORS["pink"], COLORS["purple"],
    COLORS["red"],
]

for idx, (name, complexity, data_req, note) in enumerate(methods):
    fig_family.add_trace(go.Scatter(
        x=[complexity],
        y=[data_req],
        mode="markers+text",
        marker=dict(size=18, color=method_colors[idx], opacity=0.85),
        text=[name],
        textposition="top center",
        textfont=dict(size=12, color=method_colors[idx]),
        name=name,
        hovertemplate=f"<b>{name}</b><br>{note}<br>"
                      f"Pipeline complexity: {complexity:.1f}<br>"
                      f"Data requirement: {data_req:.1f}<extra></extra>",
        showlegend=False,
    ))

fig_family.update_layout(
    title="DPO Family: Pipeline Complexity vs Data Requirements",
    xaxis_title="Pipeline Complexity (lower = simpler)",
    yaxis_title="Data Requirements (labeled preference data)",
    height=420,
    xaxis=dict(range=[0, 5]),
    yaxis=dict(range=[0, 5.5]),
)
st.plotly_chart(fig_family, use_container_width=True)

st.markdown("---")

# =====================================================================
# 6. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.1rem; padding:22px 26px;">'
    "<strong>Key Insight:</strong> DPO does not learn a reward model -- "
    "it <em>becomes</em> the reward model. The log-probability ratio "
    "pi_theta(y|x) / pi_ref(y|x) is the implicit reward, and every "
    "gradient step simultaneously reshapes both the policy and the reward "
    "landscape. This elegance is also its fragility: without the explicit "
    "reward signal as a guardrail, DPO relies entirely on the quality and "
    "freshness of its preference data."
    "</div>",
    unsafe_allow_html=True,
)
