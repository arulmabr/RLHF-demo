"""
Page 12 -- Multi-Turn & Agentic Post-Training + The Evaluation Crisis
Sections X and XI of the post-training foundations platform.
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
st.title("Multi-Turn & Agentic Post-Training")

st.markdown(
    """
    Single-turn RLHF optimizes one response at a time. But real-world use is
    **multi-turn**: a user asks a clarifying question, the model calls a tool,
    the tool returns a result, the model reasons over it, and so on. This
    creates a fundamentally harder optimization problem.
    """
)

st.markdown("---")

# =====================================================================
# 1. THE MULTI-TURN PROBLEM
# =====================================================================
st.markdown(
    '<p class="section-header">THE CORE CHALLENGE</p>',
    unsafe_allow_html=True,
)
st.subheader("The Multi-Turn Problem")

st.markdown(
    """
    Multi-turn interactions introduce four interrelated challenges that
    single-turn RLHF never has to face:
    """
)

challenges = [
    (
        "Delayed Reward",
        "A bad clarifying question in turn 2 may not cause a visible failure "
        "until turn 5. The reward signal arrives many steps after the action "
        "that caused it, making credit assignment extremely difficult.",
        COLORS["red"],
    ),
    (
        "Credit Assignment Across Turns",
        "When a five-turn conversation fails, which turn was responsible? "
        "Was it the initial framing, a missed follow-up, or a final "
        "hallucination? Decomposing the reward across turns is an open "
        "research problem akin to temporal credit assignment in RL.",
        COLORS["orange"],
    ),
    (
        "Massive Action Space",
        "At every turn the model chooses from its entire vocabulary for every "
        "position in the sequence. The total space is "
        "vocabulary^(sequence_length) per turn, and this compounds across "
        "turns. The combinatorial explosion dwarfs even the hardest board games.",
        COLORS["purple"],
    ),
    (
        "User Simulation",
        "To train multi-turn policies with RL you need a simulated user that "
        "responds realistically. But building a good user simulator is itself "
        "a hard language modeling problem -- you need a model to train a model.",
        COLORS["cyan"],
    ),
]

cols = st.columns(2)
for i, (title, desc, color) in enumerate(challenges):
    with cols[i % 2]:
        st.markdown(
            f'<div class="concept-card" style="border-left: 4px solid {color}; '
            f'min-height: 180px;">'
            f'<strong style="color:{color};">{title}</strong><br><br>'
            f'{desc}</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="insight-box">'
    "<strong>The key tension:</strong> Single-turn RLHF treats each response "
    "independently, but in a conversation, the quality of turn N depends on "
    "the entire history of turns 1 through N-1. Optimizing myopically "
    "(turn-by-turn) misses the forest for the trees."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# Delayed reward visualization
st.markdown("**Delayed Reward Visualization**")
st.markdown(
    "The diagram below shows how a poor action early in a conversation "
    "propagates forward, but the negative reward only arrives at the end."
)

num_turns_viz = 6
turn_labels = [f"Turn {i+1}" for i in range(num_turns_viz)]
# Simulate quality trajectory: a bad turn 2 causes cascading degradation
quality_good = [0.9, 0.85, 0.88, 0.90, 0.87, 0.92]
quality_bad = [0.9, 0.40, 0.65, 0.50, 0.35, 0.20]

fig_delay = go.Figure()
fig_delay.add_trace(go.Scatter(
    x=turn_labels, y=quality_good,
    mode="lines+markers",
    name="Good trajectory",
    line=dict(color=COLORS["green"], width=3),
    marker=dict(size=10),
))
fig_delay.add_trace(go.Scatter(
    x=turn_labels, y=quality_bad,
    mode="lines+markers",
    name="Bad question at turn 2",
    line=dict(color=COLORS["red"], width=3, dash="dash"),
    marker=dict(size=10),
))
fig_delay.add_annotation(
    x="Turn 2", y=0.40,
    text="Bad clarifying<br>question here",
    showarrow=True, arrowhead=2,
    font=dict(color=COLORS["red"], size=12),
    arrowcolor=COLORS["red"],
    ax=0, ay=-50,
)
fig_delay.add_annotation(
    x="Turn 6", y=0.20,
    text="Failure visible<br>only here",
    showarrow=True, arrowhead=2,
    font=dict(color=COLORS["red"], size=12),
    arrowcolor=COLORS["red"],
    ax=0, ay=-50,
)
fig_delay.update_layout(
    title="Delayed Reward: Early Mistakes Surface Late",
    yaxis=dict(title="Conversation Quality", range=[0, 1.05]),
    xaxis=dict(title=""),
    height=400,
    legend=dict(x=0.55, y=0.95),
)
st.plotly_chart(fig_delay, use_container_width=True)

st.markdown("---")

# =====================================================================
# 2. MULTI-TURN ACTION SPACE CALCULATOR
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Multi-Turn Action Space Calculator")

st.markdown(
    "Use the sliders below to see how the action space explodes as "
    "vocabulary, sequence length, and number of turns increase. We compare "
    "with the action spaces of Go and chess to put the numbers in perspective."
)

col_v, col_s, col_t = st.columns(3)
with col_v:
    vocab_size = st.slider(
        "Vocabulary size",
        min_value=1_000,
        max_value=100_000,
        value=32_000,
        step=1_000,
        format="%d",
        help="Typical range: 32K (LLaMA) to 100K (GPT-4).",
    )
with col_s:
    avg_seq_len = st.slider(
        "Avg sequence length per turn",
        min_value=100,
        max_value=10_000,
        value=500,
        step=100,
        format="%d",
        help="Average number of tokens the model generates per turn.",
    )
with col_t:
    num_turns = st.slider(
        "Number of turns",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="How many turns the model takes in the conversation.",
    )

# Compute log10 of action spaces for comparison
# Per-turn action space: vocab_size ^ avg_seq_len
# Total: (vocab_size ^ avg_seq_len) ^ num_turns = vocab_size ^ (avg_seq_len * num_turns)
log10_per_turn = avg_seq_len * np.log10(vocab_size)
log10_total = num_turns * log10_per_turn

# Reference games
log10_chess = 120.0  # Shannon number ~10^120
log10_go = 361 * np.log10(361)  # ~10^923

# Display metrics
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(
        "Per-turn action space",
        f"10^{log10_per_turn:,.0f}",
        help="vocab_size ^ avg_sequence_length",
    )
with col_m2:
    st.metric(
        "Total multi-turn action space",
        f"10^{log10_total:,.0f}",
        help="(vocab_size ^ avg_sequence_length) ^ num_turns",
    )

# Comparison bar chart (log scale)
spaces = {
    "Chess": log10_chess,
    "Go": log10_go,
    "LLM (1 turn)": log10_per_turn,
    f"LLM ({num_turns} turns)": log10_total,
}

bar_colors = [COLORS["gray"], COLORS["gray"], COLORS["blue"], COLORS["red"]]

fig_space = go.Figure(
    data=[
        go.Bar(
            x=list(spaces.keys()),
            y=list(spaces.values()),
            marker_color=bar_colors,
            text=[f"10^{v:,.0f}" for v in spaces.values()],
            textposition="outside",
            textfont=dict(size=13),
        )
    ]
)
fig_space.update_layout(
    title="Action Space Comparison (log10 scale)",
    yaxis=dict(
        title="log10(action space size)",
        type="log" if max(spaces.values()) > 10000 else "linear",
    ),
    height=420,
    showlegend=False,
)
st.plotly_chart(fig_space, use_container_width=True)

# Growth curve across turns
turns_range = np.arange(1, 21)
log10_growth = turns_range * log10_per_turn

fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(
    x=turns_range, y=log10_growth,
    mode="lines+markers",
    name="LLM action space",
    line=dict(color=COLORS["purple"], width=3),
    marker=dict(size=6),
    fill="tozeroy",
    fillcolor="rgba(155, 89, 182, 0.15)",
))
fig_growth.add_hline(
    y=log10_chess, line_dash="dash", line_color=COLORS["gray"],
    annotation_text="Chess (~10^120)",
    annotation_position="top left",
    annotation_font=dict(color=COLORS["gray"]),
)
fig_growth.add_hline(
    y=log10_go, line_dash="dash", line_color=COLORS["orange"],
    annotation_text=f"Go (~10^{log10_go:.0f})",
    annotation_position="top left",
    annotation_font=dict(color=COLORS["orange"]),
)
# Mark the selected number of turns
fig_growth.add_trace(go.Scatter(
    x=[num_turns], y=[log10_total],
    mode="markers",
    marker=dict(size=14, color=COLORS["red"], symbol="star"),
    name=f"Selected ({num_turns} turns)",
    showlegend=True,
))
fig_growth.update_layout(
    title="Action Space Explosion Across Turns",
    xaxis=dict(title="Number of Turns", dtick=2),
    yaxis=dict(title="log10(action space size)"),
    height=420,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_growth, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Scale context:</strong> Chess has roughly 10^120 possible games. "
    f"Go has roughly 10^{log10_go:.0f}. A language model with a {vocab_size:,}-token "
    f"vocabulary generating {avg_seq_len:,} tokens per turn across {num_turns} turns "
    f"has an action space of roughly 10^{log10_total:,.0f}. Even a single turn "
    "already dwarfs any board game ever studied."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. TOOL USE POST-TRAINING
# =====================================================================
st.markdown(
    '<p class="section-header">AGENTIC CAPABILITIES</p>',
    unsafe_allow_html=True,
)
st.subheader("Tool Use Post-Training")

st.markdown(
    """
    Modern language models are not just chat engines -- they are **agents**
    that call tools: search APIs, code interpreters, calculators, databases.
    The interaction follows a specific pattern:
    """
)

# Tool-use flow diagram using Plotly
fig_tool = go.Figure()

# Boxes for each step
steps = [
    ("User Query", 0.0, COLORS["blue"]),
    ("Model Thinks", 1.0, COLORS["purple"]),
    ("Tool Call\nGenerated", 2.0, COLORS["orange"]),
    ("Tool Executes\n& Returns", 3.0, COLORS["cyan"]),
    ("Model Continues\nWith Result", 4.0, COLORS["green"]),
    ("Final Answer", 5.0, COLORS["blue"]),
]

for label, x, color in steps:
    fig_tool.add_trace(go.Scatter(
        x=[x], y=[0.5],
        mode="markers+text",
        marker=dict(size=55, color=color, opacity=0.25, symbol="square"),
        text=[label],
        textposition="middle center",
        textfont=dict(size=10, color=COLORS["white"]),
        showlegend=False,
        hoverinfo="skip",
    ))

# Arrows between steps
for i in range(len(steps) - 1):
    fig_tool.add_annotation(
        x=steps[i + 1][1] - 0.15, y=0.5,
        ax=steps[i][1] + 0.15, ay=0.5,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.5, arrowwidth=2,
        arrowcolor=COLORS["gray"],
    )

fig_tool.update_layout(
    height=180,
    xaxis=dict(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[-0.5, 5.5],
    ),
    yaxis=dict(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[0, 1],
    ),
    margin=dict(l=10, r=10, t=10, b=10),
    title=None,
)
st.plotly_chart(fig_tool, use_container_width=True)

st.markdown("**Three key insights for tool-use RL:**")

insights = [
    (
        "Reward the Outcome, Not the Tool Call",
        "If you reward the model for merely generating a tool call, it will "
        "learn to call tools gratuitously. Instead, reward the **final answer "
        "quality**. The model should learn that tool calls are instrumental, "
        "not intrinsically good.",
        COLORS["green"],
        "1",
    ),
    (
        "The Exploration Problem",
        "If the model has never tried using a tool during training, it never "
        "receives the reward signal that would teach it to use that tool. This "
        "is a classic RL exploration challenge: you cannot learn the value of "
        "an action you have never taken. Without targeted exploration, the "
        "model gets stuck in a tool-free local optimum.",
        COLORS["orange"],
        "2",
    ),
    (
        "Bootstrap with SFT, Optimize with RL",
        "The practical solution: first teach the model the **syntax and basic "
        "usage** of tool calls via supervised fine-tuning on curated examples. "
        "This gets the model into the basin of attraction where tool calls are "
        "at least attempted. Then use RL to optimize **when and how** to call "
        "tools for maximum downstream reward.",
        COLORS["cyan"],
        "3",
    ),
]

for title, desc, color, num in insights:
    st.markdown(
        f'<div class="concept-card" style="border-left: 4px solid {color};">'
        f'<strong style="color:{color};">Insight {num}: {title}</strong>'
        f'<br><br>{desc}</div>',
        unsafe_allow_html=True,
    )

# Tool use training pipeline visualization
st.markdown("")
st.markdown("**Tool-Use Training Pipeline**")

stages = ["Pre-trained\nModel", "SFT on\nTool Examples", "RL with\nOutcome Reward", "Agentic\nModel"]
stage_x = [0, 1, 2, 3]
stage_colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
stage_desc = [
    "No tool use ability",
    "Learns tool syntax\n& basic patterns",
    "Learns when & how\nto call tools",
    "Effective tool\nselection & chaining",
]

fig_pipeline = go.Figure()

# Stage markers
fig_pipeline.add_trace(go.Scatter(
    x=stage_x, y=[1.0] * 4,
    mode="markers+text",
    marker=dict(size=60, color=stage_colors, opacity=0.3, symbol="square"),
    text=stages,
    textposition="middle center",
    textfont=dict(size=10, color=COLORS["white"]),
    showlegend=False,
    hoverinfo="skip",
))

# Description text below
fig_pipeline.add_trace(go.Scatter(
    x=stage_x, y=[0.5] * 4,
    mode="text",
    text=stage_desc,
    textposition="middle center",
    textfont=dict(size=9, color=COLORS["gray"]),
    showlegend=False,
    hoverinfo="skip",
))

# Arrows
for i in range(len(stages) - 1):
    fig_pipeline.add_annotation(
        x=stage_x[i + 1] - 0.2, y=1.0,
        ax=stage_x[i] + 0.2, ay=1.0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.5, arrowwidth=2,
        arrowcolor=COLORS["white"],
    )

fig_pipeline.update_layout(
    height=180,
    xaxis=dict(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[-0.5, 3.5],
    ),
    yaxis=dict(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[0, 1.5],
    ),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_pipeline, use_container_width=True)

st.markdown("---")

# =====================================================================
# 4. THE EVAL CRISIS (SECTION XI)
# =====================================================================
st.markdown(
    '<p class="section-header">SECTION XI</p>',
    unsafe_allow_html=True,
)
st.title("The Evaluation Crisis")

st.markdown(
    """
    As models get better, measuring that improvement gets harder. The field
    faces a growing **evaluation crisis**: our benchmarks are saturating,
    our judges are biased, and our most important capabilities are the hardest
    to measure.
    """
)

st.markdown("---")

st.markdown(
    '<p class="section-header">BENCHMARK LANDSCAPE</p>',
    unsafe_allow_html=True,
)
st.subheader("Major Benchmarks and Their Limitations")

st.markdown(
    "Every benchmark is a lossy proxy for the capability we actually care "
    "about. Below is an interactive overview of prominent benchmarks, what "
    "they measure, and where they fall short."
)

# Benchmark data
benchmarks = [
    {
        "name": "MT-Bench",
        "category": "Conversational",
        "measures": "Multi-turn conversation quality",
        "method": "GPT-4 as judge scores 1-10",
        "limitations": "Ceiling effects (top models all score ~9); judge bias toward GPT-4-like outputs; limited domain coverage",
        "saturation": 0.85,
        "color": COLORS["blue"],
    },
    {
        "name": "AlpacaEval",
        "category": "Instruction Following",
        "measures": "Instruction following quality",
        "method": "Win-rate vs reference model (GPT-4 judge)",
        "limitations": "Biased toward verbose, GPT-4-style responses; length gaming inflates scores; single-turn only",
        "saturation": 0.75,
        "color": COLORS["purple"],
    },
    {
        "name": "Chatbot Arena",
        "category": "Open-ended",
        "measures": "General preference via crowd-sourced A/B tests",
        "method": "Elo rating from human pairwise comparisons",
        "limitations": "English-centric and tech-heavy user base; popularity effects; demographic bias in raters",
        "saturation": 0.40,
        "color": COLORS["green"],
    },
    {
        "name": "GPQA",
        "category": "Expert Knowledge",
        "measures": "Graduate-level science questions",
        "method": "Multiple choice with expert-validated answers",
        "limitations": "Verifiable but rapidly saturating; narrow domain coverage; memorization risk",
        "saturation": 0.70,
        "color": COLORS["orange"],
    },
    {
        "name": "MATH",
        "category": "Mathematical Reasoning",
        "measures": "Competition math problem solving",
        "method": "Exact match on final numerical answer",
        "limitations": "Verifiable but near-saturated; does not test mathematical creativity or proof writing",
        "saturation": 0.90,
        "color": COLORS["red"],
    },
    {
        "name": "HumanEval",
        "category": "Code Generation",
        "measures": "Function-level code synthesis",
        "method": "pass@k on unit tests",
        "limitations": "Simple function-level tasks only; saturated at top; does not test architecture or debugging",
        "saturation": 0.92,
        "color": COLORS["cyan"],
    },
]

# Interactive benchmark selector
selected_bench = st.selectbox(
    "Select a benchmark to explore",
    [b["name"] for b in benchmarks],
    index=0,
)

b = next(bm for bm in benchmarks if bm["name"] == selected_bench)

col_info, col_sat = st.columns([2, 1])

with col_info:
    st.markdown(
        f'<div class="concept-card" style="border-left: 4px solid {b["color"]};">'
        f'<strong style="color:{b["color"]}; font-size: 1.15rem;">{b["name"]}</strong>'
        f'<br><span style="color:{COLORS["gray"]};">{b["category"]}</span>'
        f'<br><br><strong>What it measures:</strong> {b["measures"]}'
        f'<br><br><strong>Method:</strong> {b["method"]}'
        f'<br><br><strong style="color:{COLORS["orange"]};">Limitations:</strong> '
        f'{b["limitations"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_sat:
    st.markdown(
        f'<div class="concept-card" style="text-align: center;">'
        f'<strong>Saturation Level</strong><br><br>'
        f'<span style="font-size: 2.5rem; color:{b["color"]};">'
        f'{b["saturation"]:.0%}</span><br>'
        f'<span style="color:{COLORS["gray"]}; font-size: 0.85rem;">'
        f'{"Near ceiling" if b["saturation"] > 0.8 else "Saturating" if b["saturation"] > 0.6 else "Still discriminative"}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

# Saturation overview chart
fig_sat = go.Figure()

bench_names = [bm["name"] for bm in benchmarks]
saturations = [bm["saturation"] for bm in benchmarks]
bench_colors = [bm["color"] for bm in benchmarks]

fig_sat.add_trace(go.Bar(
    x=bench_names,
    y=saturations,
    marker_color=bench_colors,
    text=[f"{s:.0%}" for s in saturations],
    textposition="outside",
    textfont=dict(size=12),
))

# Danger zone
fig_sat.add_hline(
    y=0.85, line_dash="dash", line_color=COLORS["red"],
    annotation_text="Saturation danger zone",
    annotation_position="top left",
    annotation_font=dict(color=COLORS["red"], size=11),
)

fig_sat.update_layout(
    title="Benchmark Saturation Levels",
    yaxis=dict(title="Approx. Saturation", range=[0, 1.1], tickformat=".0%"),
    height=400,
    showlegend=False,
)
st.plotly_chart(fig_sat, use_container_width=True)

# Goodhart's Law visualization
st.markdown("")
st.subheader("Goodhart's Law in Action")

st.markdown(
    """
    > *When a measure becomes a target, it ceases to be a good measure.*

    As teams optimize directly for benchmark scores, the correlation between
    benchmark performance and real-world usefulness breaks down.
    """
)

# Simulate Goodhart effect
np.random.seed(42)
optimization_steps = np.arange(0, 100)
benchmark_score = 1 - np.exp(-optimization_steps / 20)  # monotonically increases
# True capability rises, plateaus, then can actually decrease
true_capability = (
    0.7 * (1 - np.exp(-optimization_steps / 25))
    - 0.15 * np.maximum(0, (optimization_steps - 50) / 50) ** 1.5
)
true_capability = np.clip(true_capability, 0, 1)

# Correlation over time
window = 15
correlation = []
for i in range(len(optimization_steps)):
    start = max(0, i - window)
    if i - start < 3:
        correlation.append(1.0)
    else:
        c = np.corrcoef(
            benchmark_score[start:i + 1],
            true_capability[start:i + 1]
        )[0, 1]
        correlation.append(c if not np.isnan(c) else correlation[-1])
correlation = np.array(correlation)

fig_goodhart = go.Figure()

fig_goodhart.add_trace(go.Scatter(
    x=optimization_steps, y=benchmark_score,
    mode="lines",
    name="Benchmark Score",
    line=dict(color=COLORS["blue"], width=3),
))
fig_goodhart.add_trace(go.Scatter(
    x=optimization_steps, y=true_capability,
    mode="lines",
    name="True Capability",
    line=dict(color=COLORS["green"], width=3),
))
fig_goodhart.add_trace(go.Scatter(
    x=optimization_steps, y=correlation,
    mode="lines",
    name="Correlation",
    line=dict(color=COLORS["orange"], width=2, dash="dot"),
    yaxis="y2",
))

# Mark the divergence point
diverge_idx = np.argmax(true_capability)
fig_goodhart.add_vline(
    x=diverge_idx, line_dash="dash", line_color=COLORS["red"],
    annotation_text="Goodhart divergence",
    annotation_position="top",
    annotation_font=dict(color=COLORS["red"], size=11),
)

fig_goodhart.update_layout(
    title="Goodhart's Law: When the Benchmark Becomes the Target",
    xaxis=dict(title="Optimization Pressure (training steps)"),
    yaxis=dict(title="Score / Capability", range=[0, 1.1]),
    yaxis2=dict(
        title="Correlation",
        overlaying="y",
        side="right",
        range=[-0.2, 1.1],
        showgrid=False,
    ),
    height=450,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_goodhart, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Goodhart's Law</strong> is not just a theoretical concern. "
    "Teams have been caught gaming AlpacaEval by training models to be "
    "verbose (longer responses get higher GPT-4 judge scores). MT-Bench "
    "ceiling effects mean that genuinely different models get indistinguishable "
    "scores. Once a benchmark becomes a target, it stops measuring what you "
    "care about."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 5. EVALUATING SUBJECTIVE CAPABILITIES
# =====================================================================
st.markdown(
    '<p class="section-header">THE HARDEST FRONTIER</p>',
    unsafe_allow_html=True,
)
st.subheader("Evaluating Subjective Capabilities")

st.markdown(
    """
    The most important capabilities for real-world usefulness are also the
    hardest to evaluate: **emotional intelligence**, **creativity**,
    **narrative ability**, and **taste**. These lack verifiable ground truth
    -- there is no "correct answer" to check against.
    """
)

# Capability difficulty spectrum
capabilities = [
    ("Arithmetic", 0.05, "Exact match", COLORS["green"]),
    ("Code correctness", 0.15, "Unit tests", COLORS["green"]),
    ("Factual QA", 0.25, "Reference answers", COLORS["green"]),
    ("Logical reasoning", 0.40, "Proof checking", COLORS["blue"]),
    ("Summarization", 0.55, "Rubric + human", COLORS["orange"]),
    ("Persuasiveness", 0.65, "Human judgment", COLORS["orange"]),
    ("Emotional intelligence", 0.75, "Expert panel", COLORS["red"]),
    ("Creative writing", 0.82, "Subjective taste", COLORS["red"]),
    ("Narrative coherence", 0.88, "Long-form eval", COLORS["red"]),
    ("Aesthetic taste", 0.95, "No consensus", COLORS["red"]),
]

cap_names = [c[0] for c in capabilities]
cap_difficulty = [c[1] for c in capabilities]
cap_methods = [c[2] for c in capabilities]
cap_colors = [c[3] for c in capabilities]

fig_spectrum = go.Figure()

fig_spectrum.add_trace(go.Bar(
    y=cap_names,
    x=cap_difficulty,
    orientation="h",
    marker_color=cap_colors,
    text=cap_methods,
    textposition="outside",
    textfont=dict(size=11),
))

fig_spectrum.update_layout(
    title="Evaluation Difficulty Spectrum",
    xaxis=dict(
        title="Evaluation Difficulty",
        range=[0, 1.25],
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=["Easy\n(verifiable)", "", "Medium\n(rubric-based)", "", "Hard\n(subjective)"],
    ),
    yaxis=dict(autorange="reversed"),
    height=450,
    showlegend=False,
    margin=dict(l=150, r=80, t=50, b=60),
)
st.plotly_chart(fig_spectrum, use_container_width=True)

st.markdown("**Current approaches to subjective evaluation:**")

approaches = [
    (
        "Rubric-Based Human Evaluation",
        "Define explicit scoring rubrics (e.g., 1-5 scales for helpfulness, "
        "accuracy, harmlessness) and have trained annotators rate outputs. "
        "Advantages: reproducible, calibrated. Disadvantages: rubrics cannot "
        "capture everything, inter-annotator agreement is often low.",
        COLORS["blue"],
    ),
    (
        "Pairwise Comparison by Experts",
        "Show two model outputs side-by-side to domain experts and ask which "
        "is better. This is the basis of Chatbot Arena. Advantages: simpler "
        "judgment than absolute scoring, surfaces subtle differences. "
        "Disadvantages: expensive, slow, does not scale, no absolute quality "
        "signal.",
        COLORS["purple"],
    ),
    (
        "Long-Form Interaction Quality",
        "Evaluate models through extended multi-turn conversations with expert "
        "users who rate the overall experience. Advantages: captures what "
        "actually matters in deployment. Disadvantages: extremely expensive, "
        "hard to standardize, results are noisy.",
        COLORS["cyan"],
    ),
]

for title, desc, color in approaches:
    st.markdown(
        f'<div class="concept-card" style="border-left: 4px solid {color};">'
        f'<strong style="color:{color};">{title}</strong>'
        f'<br><br>{desc}</div>',
        unsafe_allow_html=True,
    )

# Tradeoff visualization
st.markdown("")
st.markdown("**Evaluation Method Tradeoffs**")

eval_methods = [
    ("Exact match\n(MATH, code)", 0.95, 0.15, 0.10, COLORS["green"]),
    ("LLM-as-judge\n(MT-Bench)", 0.65, 0.55, 0.60, COLORS["blue"]),
    ("Rubric + human\n(Summarization)", 0.50, 0.75, 0.75, COLORS["orange"]),
    ("Pairwise expert\n(Arena)", 0.30, 0.85, 0.85, COLORS["purple"]),
    ("Long-form\ninteraction", 0.15, 0.95, 0.95, COLORS["red"]),
]

fig_tradeoff = go.Figure()

for name, scalability, expressiveness, cost, color in eval_methods:
    fig_tradeoff.add_trace(go.Scatter(
        x=[scalability],
        y=[expressiveness],
        mode="markers+text",
        marker=dict(size=cost * 40 + 15, color=color, opacity=0.7),
        text=[name],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["white"]),
        name=name.replace("\n", " "),
        showlegend=False,
        hovertemplate=(
            f"<b>{name.replace(chr(10), ' ')}</b><br>"
            f"Scalability: {scalability:.0%}<br>"
            f"Expressiveness: {expressiveness:.0%}<br>"
            f"Relative cost: {cost:.0%}<extra></extra>"
        ),
    ))

fig_tradeoff.update_layout(
    title="Evaluation Method Tradeoffs (bubble size = relative cost)",
    xaxis=dict(title="Scalability", range=[-0.05, 1.1], tickformat=".0%"),
    yaxis=dict(title="Expressiveness", range=[-0.05, 1.15], tickformat=".0%"),
    height=450,
)

# Quadrant annotations
fig_tradeoff.add_annotation(
    x=0.85, y=0.05,
    text="Cheap but<br>shallow",
    showarrow=False,
    font=dict(color=COLORS["gray"], size=10),
)
fig_tradeoff.add_annotation(
    x=0.10, y=1.05,
    text="Rich but<br>expensive",
    showarrow=False,
    font=dict(color=COLORS["gray"], size=10),
)

st.plotly_chart(fig_tradeoff, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>The fundamental tradeoff:</strong> The evaluation methods that "
    "best capture what we care about (long-form expert interaction) are the "
    "least scalable. The methods that scale (exact match, LLM-as-judge) "
    "capture only a fraction of real capability. There is no free lunch in "
    "evaluation."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 6. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.1rem; padding:24px 28px; '
    f'border-left: 5px solid {COLORS["red"]};">'
    f'<strong style="color:{COLORS["red"]};">Key Insight:</strong> '
    "Arguably the most important unsolved problem in post-training is "
    "evaluation. The better the model, the harder it is to measure. "
    "Benchmarks saturate, judges are biased, and our most valued "
    "capabilities -- creativity, emotional depth, taste -- resist "
    "quantification entirely. Building better evaluation is not a side "
    "quest; it is the bottleneck for the entire field."
    "</div>",
    unsafe_allow_html=True,
)
