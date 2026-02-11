"""
Page 13 -- Frontier: The Compression Frame
Every technique = a different way to compress human values into a neural network.
"""

from style import inject_custom_css, COLORS
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">FRONTIER</p>',
    unsafe_allow_html=True,
)
st.title("The Compression Frame")

st.markdown(
    """
    Every post-training technique is, at its core, a **different way to compress
    human values into a neural network**. The techniques differ in *what* they
    compress, *how lossy* the compression is, and *where* they break down.

    This page synthesizes the entire course through that single lens.
    """
)

st.markdown("---")

# =====================================================================
# 1. THE COMPRESSION FRAME TABLE
# =====================================================================
st.markdown(
    '<p class="section-header">SYNTHESIS</p>',
    unsafe_allow_html=True,
)
st.subheader("The Compression Frame")
st.markdown(
    "Each method compresses a different facet of human intent. "
    "The question is always: **what is lost?**"
)

methods_data = [
    {
        "method": "SFT",
        "what_compressed": "Demonstrated behaviors",
        "compression_type": "Behavioral cloning",
        "loss_character": "Lossy -- only captures surface patterns, not underlying intent",
        "key_failure": "Mimics form without understanding function",
        "color": COLORS["blue"],
    },
    {
        "method": "Reward Modeling",
        "what_compressed": "Human preferences",
        "compression_type": "Preference distillation",
        "loss_character": "Lossy -- Goodhart's law is the compression artifact",
        "key_failure": "Reward hacking exploits the gap between proxy and true reward",
        "color": COLORS["red"],
    },
    {
        "method": "RLHF",
        "what_compressed": "Preferences via RL optimization",
        "compression_type": "Policy optimization against reward proxy",
        "loss_character": "Powerful but unstable -- KL penalty trades off fidelity vs. exploitation",
        "key_failure": "Mode collapse, reward hacking at high KL budgets",
        "color": COLORS["green"],
    },
    {
        "method": "DPO",
        "what_compressed": "Preferences directly into policy weights",
        "compression_type": "Closed-form policy update",
        "loss_character": "Simpler, but offline data limits adaptation",
        "key_failure": "Cannot explore beyond the preference dataset distribution",
        "color": COLORS["orange"],
    },
    {
        "method": "Constitutional AI",
        "what_compressed": "Preferences via abstract principles",
        "compression_type": "Principle-guided self-critique",
        "loss_character": "Elegant and scalable, but principles have gaps",
        "key_failure": "Principles can conflict; coverage is never complete",
        "color": COLORS["purple"],
    },
    {
        "method": "RL on Verifiable Rewards",
        "what_compressed": "Objective correctness",
        "compression_type": "Lossless (within domain)",
        "loss_character": "Lossless -- but only where ground truth exists",
        "key_failure": "Limited to domains with verifiable answers",
        "color": COLORS["cyan"],
    },
    {
        "method": "Best-of-N Sampling",
        "what_compressed": "Quality via rejection sampling",
        "compression_type": "Inference-time search",
        "loss_character": "Simple but logarithmic cost scaling",
        "key_failure": "Reward improves as O(log N) -- diminishing returns",
        "color": COLORS["pink"],
    },
]

# Build the interactive table with colored method badges
table_html = """
<div style="overflow-x: auto;">
<table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
<thead>
<tr style="border-bottom: 2px solid {grid};">
    <th style="padding: 12px 16px; text-align: left; color: {white};">Method</th>
    <th style="padding: 12px 16px; text-align: left; color: {white};">What is Compressed</th>
    <th style="padding: 12px 16px; text-align: left; color: {white};">Compression Type</th>
    <th style="padding: 12px 16px; text-align: left; color: {white};">Loss Character</th>
    <th style="padding: 12px 16px; text-align: left; color: {white};">Key Failure Mode</th>
</tr>
</thead>
<tbody>
""".format(grid=COLORS["grid"], white=COLORS["white"])

for m in methods_data:
    table_html += """
<tr style="border-bottom: 1px solid {grid};">
    <td style="padding: 10px 16px;">
        <span style="background: {color}22; color: {color}; padding: 4px 10px;
        border-radius: 6px; font-weight: 600; white-space: nowrap;">{method}</span>
    </td>
    <td style="padding: 10px 16px; color: {white};">{what}</td>
    <td style="padding: 10px 16px; color: {gray};">{ctype}</td>
    <td style="padding: 10px 16px; color: {white};">{loss}</td>
    <td style="padding: 10px 16px; color: {gray}; font-style: italic;">{failure}</td>
</tr>
""".format(
        grid=COLORS["grid"],
        color=m["color"],
        method=m["method"],
        what=m["what_compressed"],
        ctype=m["compression_type"],
        loss=m["loss_character"],
        failure=m["key_failure"],
        white=COLORS["white"],
        gray=COLORS["gray"],
    )

table_html += "</tbody></table></div>"
st.markdown(table_html, unsafe_allow_html=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The pattern:</strong> Every method trades off between what it can "
    "faithfully compress and what it distorts or discards. Goodhart's law, mode "
    "collapse, distribution limits -- these are all <em>compression artifacts</em>."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 2. METHOD COMPARISON RADAR CHART
# =====================================================================
st.markdown(
    '<p class="section-header">COMPARISON</p>',
    unsafe_allow_html=True,
)
st.subheader("Method Comparison Radar Chart")
st.markdown(
    "Compare post-training methods across six key dimensions. "
    "Select the methods you want to overlay."
)

# Dimensions for the radar chart
dimensions = [
    "Implementation\nComplexity",
    "Compute\nCost",
    "Data\nRequirements",
    "KL Efficiency\n(reward/nat)",
    "Stability",
    "Scalability",
]

# Scores for each method on each dimension (0-10 scale, higher = better)
radar_data = {
    "SFT": [2, 3, 5, 4, 9, 6],
    "Reward Modeling": [6, 5, 7, 6, 7, 7],
    "RLHF": [9, 8, 7, 9, 3, 8],
    "DPO": [4, 3, 6, 7, 8, 5],
    "Constitutional AI": [7, 6, 3, 7, 6, 9],
    "RL on Verifiable": [5, 4, 2, 10, 7, 7],
    "Best-of-N": [1, 7, 1, 3, 10, 3],
}

radar_colors = {
    "SFT": COLORS["blue"],
    "Reward Modeling": COLORS["red"],
    "RLHF": COLORS["green"],
    "DPO": COLORS["orange"],
    "Constitutional AI": COLORS["purple"],
    "RL on Verifiable": COLORS["cyan"],
    "Best-of-N": COLORS["pink"],
}

selected_methods = st.multiselect(
    "Select methods to compare",
    list(radar_data.keys()),
    default=["SFT", "RLHF", "DPO", "Constitutional AI"],
)

if selected_methods:
    fig_radar = go.Figure()

    for method_name in selected_methods:
        values = radar_data[method_name]
        # Close the polygon
        values_closed = values + [values[0]]
        dims_closed = dimensions + [dimensions[0]]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=dims_closed,
                fill="toself",
                fillcolor=radar_colors[method_name] + "18",
                line=dict(color=radar_colors[method_name], width=2),
                name=method_name,
                hovertemplate="%{theta}: %{r}/10<extra>" + method_name + "</extra>",
            )
        )

    fig_radar.update_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=10, color=COLORS["gray"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=11, color=COLORS["white"]),
            ),
        ),
        height=520,
        margin=dict(l=80, r=80, t=40, b=40),
        legend=dict(
            font=dict(size=12),
            bgcolor=COLORS["card"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
        showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<strong>Reading the radar:</strong> No method dominates all dimensions. "
        "RLHF achieves the highest KL efficiency but the lowest stability. "
        "Best-of-N is trivial to implement but has poor scalability. "
        "The art of post-training is choosing the right compression for your problem."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Select at least one method above to see the radar chart.")

st.markdown("---")

# =====================================================================
# 3. SCALABLE OVERSIGHT
# =====================================================================
st.markdown(
    '<p class="section-header">THE FRONTIER QUESTION</p>',
    unsafe_allow_html=True,
)
st.subheader("Scalable Oversight")
st.markdown(
    """
    The central unsolved problem: **How do you train a model that is better than
    its teacher?**

    If post-training compresses human judgment, what happens when the model
    exceeds human judgment? Every method below attacks this problem differently.
    """
)

# Interactive diagram of scalable oversight approaches
oversight_approaches = {
    "Debate": {
        "description": "Two models argue opposing sides; a human judges which argument is stronger. "
                       "The key insight: it is easier to judge an argument than to generate one.",
        "mechanism": "Model A argues for X, Model B argues against X, Human picks the winner",
        "strength": "Leverages verification being easier than generation",
        "weakness": "Assumes humans can judge sophisticated arguments",
        "color": COLORS["blue"],
        "icon": "Debate",
    },
    "Recursive Reward Modeling": {
        "description": "Break hard tasks into easier subtasks that humans CAN evaluate. "
                       "Each level of the recursion handles one level of complexity.",
        "mechanism": "Hard task -> decompose -> human evaluates leaves -> aggregate rewards up",
        "strength": "Theoretically handles arbitrary complexity",
        "weakness": "Decomposition itself may require superhuman ability",
        "color": COLORS["green"],
        "icon": "RRM",
    },
    "Constitutional AI": {
        "description": "Principles generalize beyond the specific examples annotators could provide. "
                       "A good principle compresses more than any finite set of demonstrations.",
        "mechanism": "Principles -> self-critique -> revision -> distillation",
        "strength": "Principles can generalize to novel situations annotators never saw",
        "weakness": "Principles may be incomplete or contradictory",
        "color": COLORS["purple"],
        "icon": "CAI",
    },
    "Verification != Generation": {
        "description": "The fundamental asymmetry that makes oversight possible. "
                       "Verifying a proof is easier than discovering one. Judging code is easier than writing it.",
        "mechanism": "Model generates candidates -> verifier checks -> accept or reject",
        "strength": "Applies wherever verification is computationally easier",
        "weakness": "Not all tasks have efficient verification",
        "color": COLORS["cyan"],
        "icon": "V!=G",
    },
}

selected_approach = st.selectbox(
    "Explore an approach",
    list(oversight_approaches.keys()),
)

approach = oversight_approaches[selected_approach]

# Build the interactive diagram
st.markdown(
    '<div class="concept-card" style="border-left: 4px solid {color};">'
    '<strong style="color: {color}; font-size: 1.1rem;">{name}</strong>'
    '<p style="margin: 10px 0; color: {white};">{desc}</p>'
    '<div style="background: {bg}; border-radius: 8px; padding: 12px 16px; '
    'margin: 10px 0; font-family: monospace; font-size: 0.9rem; color: {gray};">'
    '{mechanism}'
    '</div>'
    '<div style="display: flex; gap: 16px; margin-top: 10px;">'
    '<div style="flex: 1; padding: 8px 12px; background: #2ECC7118; '
    'border-radius: 6px; border-left: 3px solid #2ECC71;">'
    '<strong style="color: #2ECC71; font-size: 0.8rem;">STRENGTH</strong><br>'
    '<span style="color: {white}; font-size: 0.85rem;">{strength}</span>'
    '</div>'
    '<div style="flex: 1; padding: 8px 12px; background: #E74C3C18; '
    'border-radius: 6px; border-left: 3px solid #E74C3C;">'
    '<strong style="color: #E74C3C; font-size: 0.8rem;">WEAKNESS</strong><br>'
    '<span style="color: {white}; font-size: 0.85rem;">{weakness}</span>'
    '</div>'
    '</div>'
    '</div>'.format(
        color=approach["color"],
        name=selected_approach,
        desc=approach["description"],
        mechanism=approach["mechanism"],
        strength=approach["strength"],
        weakness=approach["weakness"],
        white=COLORS["white"],
        gray=COLORS["gray"],
        bg=COLORS["bg"],
    ),
    unsafe_allow_html=True,
)

# Scalable oversight diagram: capability vs. oversight difficulty
st.markdown("")
st.markdown("**The Oversight Gap**")
st.markdown(
    "As model capability grows, the gap between what the model can do and "
    "what humans can directly verify widens. Scalable oversight methods attempt "
    "to close this gap."
)

capability_x = np.linspace(0, 10, 100)
model_capability = 1.0 / (1.0 + np.exp(-1.2 * (capability_x - 5))) * 10
human_direct = np.ones_like(capability_x) * 4.0
debate_line = human_direct + 1.5 * np.log1p(capability_x)
rrm_line = human_direct + 2.0 * np.log1p(capability_x)
cai_line = human_direct + 1.0 * np.log1p(capability_x) + 0.3 * capability_x

fig_oversight = go.Figure()

fig_oversight.add_trace(
    go.Scatter(
        x=capability_x,
        y=model_capability,
        mode="lines",
        name="Model capability",
        line=dict(color=COLORS["red"], width=3),
    )
)
fig_oversight.add_trace(
    go.Scatter(
        x=capability_x,
        y=human_direct,
        mode="lines",
        name="Human direct oversight",
        line=dict(color=COLORS["gray"], width=2, dash="dash"),
    )
)
fig_oversight.add_trace(
    go.Scatter(
        x=capability_x,
        y=debate_line,
        mode="lines",
        name="Debate-amplified oversight",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig_oversight.add_trace(
    go.Scatter(
        x=capability_x,
        y=rrm_line,
        mode="lines",
        name="Recursive reward modeling",
        line=dict(color=COLORS["green"], width=2),
    )
)
fig_oversight.add_trace(
    go.Scatter(
        x=capability_x,
        y=cai_line,
        mode="lines",
        name="Constitutional AI (principles)",
        line=dict(color=COLORS["purple"], width=2),
    )
)

# Add the gap annotation
fig_oversight.add_annotation(
    x=8, y=7,
    ax=8, ay=5.5,
    text="Oversight Gap",
    showarrow=True,
    arrowhead=3,
    arrowcolor=COLORS["orange"],
    font=dict(color=COLORS["orange"], size=12),
)

fig_oversight.update_layout(
    title="The Scalable Oversight Problem",
    xaxis_title="Pre-training Scale",
    yaxis_title="Capability / Oversight Reach",
    height=450,
    legend=dict(
        font=dict(size=11),
        bgcolor=COLORS["card"],
        bordercolor=COLORS["grid"],
        borderwidth=1,
    ),
    yaxis=dict(range=[0, 12]),
)
st.plotly_chart(fig_oversight, use_container_width=True)

st.markdown("---")

# =====================================================================
# 4. THE SCALING QUESTION
# =====================================================================
st.markdown(
    '<p class="section-header">OPEN QUESTION</p>',
    unsafe_allow_html=True,
)
st.subheader("Does Post-Training Scale Predictably?")
st.markdown(
    """
    Pre-training follows clean scaling laws (loss decreases as a power law with
    compute). Post-training is murkier. Some techniques might scale. Others might
    hit ceilings.
    """
)

post_train_compute = st.slider(
    "Post-training compute (relative units)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    format="%.0fx",
    help="Adjust the relative amount of post-training compute.",
)

compute_range = np.linspace(1, 100, 200)

# Categories that might scale
rl_verifiable = 0.3 * np.log(compute_range) / np.log(100) + 0.4
data_quality = 0.25 * np.log(compute_range) / np.log(100) + 0.45
multi_turn_rl = 0.35 * np.log(compute_range) / np.log(100) + 0.3

# Categories that might plateau
pref_data = 0.5 * (1 - np.exp(-compute_range / 15)) + 0.3
subjective = 0.3 * (1 - np.exp(-compute_range / 8)) + 0.25

# Compute the values at the selected compute level
def interp_val(curve, x_val):
    idx = np.searchsorted(compute_range, x_val)
    idx = min(idx, len(curve) - 1)
    return curve[idx]

col_scale_l, col_scale_r = st.columns(2)

with col_scale_l:
    st.markdown(
        '<div class="concept-card" style="border-left: 3px solid {green};">'
        '<strong style="color: {green};">Might Scale</strong>'
        '<ul style="color: {white}; margin-top: 8px;">'
        '<li>RL on verifiable rewards: <strong>{v1:.0%}</strong></li>'
        '<li>Data quality (fixed quantity): <strong>{v2:.0%}</strong></li>'
        '<li>Multi-turn RL: <strong>{v3:.0%}</strong></li>'
        '</ul></div>'.format(
            green=COLORS["green"],
            white=COLORS["white"],
            v1=interp_val(rl_verifiable, post_train_compute),
            v2=interp_val(data_quality, post_train_compute),
            v3=interp_val(multi_turn_rl, post_train_compute),
        ),
        unsafe_allow_html=True,
    )

with col_scale_r:
    st.markdown(
        '<div class="concept-card" style="border-left: 3px solid {red};">'
        '<strong style="color: {red};">Might Plateau</strong>'
        '<ul style="color: {white}; margin-top: 8px;">'
        '<li>Preference data (beyond threshold): <strong>{v1:.0%}</strong></li>'
        '<li>Subjective capability training: <strong>{v2:.0%}</strong></li>'
        '</ul></div>'.format(
            red=COLORS["red"],
            white=COLORS["white"],
            v1=interp_val(pref_data, post_train_compute),
            v2=interp_val(subjective, post_train_compute),
        ),
        unsafe_allow_html=True,
    )

fig_scaling = go.Figure()

# Might-scale curves
fig_scaling.add_trace(
    go.Scatter(
        x=compute_range, y=rl_verifiable,
        mode="lines", name="RL on verifiable rewards",
        line=dict(color=COLORS["cyan"], width=2),
    )
)
fig_scaling.add_trace(
    go.Scatter(
        x=compute_range, y=data_quality,
        mode="lines", name="Data quality (fixed quantity)",
        line=dict(color=COLORS["green"], width=2),
    )
)
fig_scaling.add_trace(
    go.Scatter(
        x=compute_range, y=multi_turn_rl,
        mode="lines", name="Multi-turn RL",
        line=dict(color=COLORS["blue"], width=2),
    )
)

# Might-plateau curves
fig_scaling.add_trace(
    go.Scatter(
        x=compute_range, y=pref_data,
        mode="lines", name="Preference data (beyond threshold)",
        line=dict(color=COLORS["orange"], width=2, dash="dash"),
    )
)
fig_scaling.add_trace(
    go.Scatter(
        x=compute_range, y=subjective,
        mode="lines", name="Subjective capability training",
        line=dict(color=COLORS["red"], width=2, dash="dash"),
    )
)

# Add a vertical line for current compute selection
fig_scaling.add_vline(
    x=post_train_compute,
    line_dash="dot",
    line_color=COLORS["yellow"],
    annotation_text=f"{post_train_compute:.0f}x compute",
    annotation_font_color=COLORS["yellow"],
)

fig_scaling.update_layout(
    title="Projected Capability Improvement vs. Post-Training Compute",
    xaxis_title="Relative Post-Training Compute",
    yaxis_title="Capability Score",
    xaxis=dict(type="log"),
    yaxis=dict(range=[0, 1.0], tickformat=".0%"),
    height=450,
    legend=dict(
        font=dict(size=11),
        bgcolor=COLORS["card"],
        bordercolor=COLORS["grid"],
        borderwidth=1,
    ),
)
st.plotly_chart(fig_scaling, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The scaling question matters</strong> because it determines where to "
    "invest. If subjective capabilities plateau, we need fundamentally new "
    "compression methods -- not just more compute on existing ones."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 5. THE SUBJECTIVE INTELLIGENCE GAP
# =====================================================================
st.markdown(
    '<p class="section-header">THE HARD PROBLEM</p>',
    unsafe_allow_html=True,
)
st.subheader("The Subjective Intelligence Gap")
st.markdown(
    """
    Current methods optimize for **consensus** -- the mean field approximation
    of human preferences. But the most valuable subjective capabilities
    (taste, emotional depth, creative vision) are **anti-consensus**.
    They are valuable *precisely because* they diverge from the average.

    Averaging destroys what makes them valuable.
    """
)

# Demonstrate how averaging destroys subjective signal
st.markdown("**How Averaging Destroys Subjective Value**")

n_annotators = st.slider(
    "Number of annotators (diversity of taste)",
    min_value=3,
    max_value=50,
    value=10,
    step=1,
)

np.random.seed(42)

# Generate diverse individual preference profiles (each annotator has unique taste)
categories = [
    "Formal\nPrecision", "Poetic\nLanguage", "Dry\nHumor",
    "Emotional\nDepth", "Provocative\nEdge", "Minimalist\nStyle",
    "Warm\nTone", "Technical\nRigor",
]

n_cats = len(categories)
theta = np.linspace(0, 2 * np.pi, n_cats, endpoint=False)

# Each annotator has a distinct preference peak
individual_profiles = []
for i in range(n_annotators):
    # Each annotator strongly prefers 1-2 categories
    profile = np.random.dirichlet(np.ones(n_cats) * 0.5) * 10
    individual_profiles.append(profile)

individual_profiles = np.array(individual_profiles)

# The consensus (mean) profile
consensus_profile = individual_profiles.mean(axis=0)

# Measure how much information is lost
individual_variances = individual_profiles.var(axis=1).mean()
consensus_variance = consensus_profile.var()
variance_ratio = consensus_variance / max(individual_variances, 1e-10)

fig_subjective = go.Figure()

# Show a few individual annotator profiles
n_show = min(5, n_annotators)
for i in range(n_show):
    vals = individual_profiles[i].tolist() + [individual_profiles[i][0]]
    cats_closed = categories + [categories[0]]
    fig_subjective.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=cats_closed,
            mode="lines",
            name=f"Annotator {i + 1}",
            line=dict(color=COLORS["gray"], width=1),
            opacity=0.4,
        )
    )

# Show the consensus profile prominently
consensus_closed = consensus_profile.tolist() + [consensus_profile[0]]
cats_closed = categories + [categories[0]]
fig_subjective.add_trace(
    go.Scatterpolar(
        r=consensus_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=COLORS["red"] + "25",
        line=dict(color=COLORS["red"], width=3),
        name="Consensus (mean)",
    )
)

fig_subjective.update_layout(
    polar=dict(
        bgcolor=COLORS["bg"],
        radialaxis=dict(
            visible=True,
            range=[0, max(individual_profiles.max(), consensus_profile.max()) + 1],
            gridcolor=COLORS["grid"],
            tickfont=dict(size=9, color=COLORS["gray"]),
        ),
        angularaxis=dict(
            gridcolor=COLORS["grid"],
            tickfont=dict(size=10, color=COLORS["white"]),
        ),
    ),
    title=f"Individual Preferences vs. Consensus ({n_annotators} annotators)",
    height=500,
    margin=dict(l=80, r=80, t=60, b=40),
    legend=dict(
        font=dict(size=10),
        bgcolor=COLORS["card"],
        bordercolor=COLORS["grid"],
        borderwidth=1,
    ),
)
st.plotly_chart(fig_subjective, use_container_width=True)

# Show the information loss metric
col_gap1, col_gap2, col_gap3 = st.columns(3)

with col_gap1:
    st.metric(
        "Avg Individual Variance",
        f"{individual_variances:.2f}",
        help="How distinctive each annotator's taste is",
    )

with col_gap2:
    st.metric(
        "Consensus Variance",
        f"{consensus_variance:.2f}",
        help="How distinctive the averaged profile is",
    )

with col_gap3:
    pct_lost = max(0, (1 - variance_ratio)) * 100
    st.metric(
        "Distinctiveness Lost",
        f"{pct_lost:.0f}%",
        delta=f"-{pct_lost:.0f}%",
        delta_color="inverse",
        help="How much individual character is destroyed by averaging",
    )

# Show bar chart of entropy per annotator vs consensus
individual_entropies = []
for i in range(n_annotators):
    p = individual_profiles[i] / individual_profiles[i].sum()
    p = np.clip(p, 1e-10, 1.0)
    individual_entropies.append(-np.sum(p * np.log2(p)))

p_consensus = consensus_profile / consensus_profile.sum()
p_consensus = np.clip(p_consensus, 1e-10, 1.0)
consensus_entropy = -np.sum(p_consensus * np.log2(p_consensus))
max_entropy = np.log2(n_cats)

fig_entropy = go.Figure()

fig_entropy.add_trace(
    go.Bar(
        x=[f"A{i+1}" for i in range(min(n_annotators, 20))],
        y=individual_entropies[:20],
        marker_color=COLORS["blue"],
        name="Individual annotators",
        opacity=0.6,
    )
)

# Add consensus line
fig_entropy.add_hline(
    y=consensus_entropy,
    line_dash="dash",
    line_color=COLORS["red"],
    annotation_text=f"Consensus entropy: {consensus_entropy:.2f}",
    annotation_font_color=COLORS["red"],
)

fig_entropy.add_hline(
    y=max_entropy,
    line_dash="dot",
    line_color=COLORS["gray"],
    annotation_text=f"Maximum entropy (uniform): {max_entropy:.2f}",
    annotation_font_color=COLORS["gray"],
    annotation_position="bottom left",
)

fig_entropy.update_layout(
    title="Preference Entropy: Individuals vs. Consensus",
    xaxis_title="Annotator",
    yaxis_title="Entropy (bits)",
    height=350,
    showlegend=False,
    yaxis=dict(range=[0, max_entropy + 0.5]),
)
st.plotly_chart(fig_entropy, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The paradox:</strong> Individual annotators have low entropy "
    "(strong, distinctive preferences). The consensus has high entropy "
    "(close to uniform -- all preferences washed out). Training on consensus "
    "produces models that are competent but bland. The most important capabilities "
    "-- taste, voice, creative vision -- are precisely what averaging destroys."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 6. KEY INSIGHT
# =====================================================================
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
    border-radius: 14px; padding: 32px 36px; margin: 20px 0;
    border: 1px solid {orange}44; text-align: center;">
    <p style="font-size: 0.8rem; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: {orange}; margin-bottom: 16px;">
    THE COMPRESSION FRAME
    </p>
    <p style="font-size: 1.25rem; color: {white}; line-height: 1.8; margin: 0;">
    You can't compress what you don't understand.<br>
    You can't teach what you can't compress.<br>
    <span style="color: {orange}; font-weight: 600;">
    The most important capabilities may be the hardest to compress.</span><br>
    <em style="color: {gray};">And that's the work.</em>
    </p>
    </div>
    """.format(
        orange=COLORS["orange"],
        white=COLORS["white"],
        gray=COLORS["gray"],
    ),
    unsafe_allow_html=True,
)
