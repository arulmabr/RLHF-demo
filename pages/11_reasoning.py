"""
Page 11 -- Reasoning: The o1/R1 Paradigm
DeepSeek R1 pipeline, GRPO, emergent behaviors, PRM, CoT as decompression.
"""

from style import inject_custom_css, COLORS
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION XI</p>',
    unsafe_allow_html=True,
)
st.title("Reasoning: The o1 / R1 Paradigm")

st.markdown(
    """
    Reasoning models learn to **think step-by-step** before producing a final
    answer.  The key insight behind o1, DeepSeek-R1, and similar systems is
    that chain-of-thought (CoT) reasoning can be *trained into* models through
    reinforcement learning on verifiable tasks -- and that doing so unlocks
    emergent problem-solving behaviors that were never explicitly taught.
    """
)

st.markdown(
    """
    **The DeepSeek R1 Pipeline** follows three phases:

    | Phase | Method | Purpose |
    |-------|--------|---------|
    | **Phase 1** | Cold-start SFT on long CoT traces | Bootstrap the model with structured reasoning format |
    | **Phase 2** | Large-scale RL with GRPO on verifiable rewards | Train the model to maximize correctness through reasoning |
    | **Phase 3** | Rejection sampling, SFT, then another round of RL | Distill best reasoning traces back, then refine further |
    """
)

st.markdown("---")

# =====================================================================
# 1. COT AS DECOMPRESSION -- EXPLANATION
# =====================================================================
st.markdown(
    '<p class="section-header">CORE IDEA</p>',
    unsafe_allow_html=True,
)
st.subheader("Chain-of-Thought as Decompression")

st.markdown(
    """
    Think of a direct answer as a **highly compressed** representation -- all the
    reasoning is hidden inside the model's weights.  Chain-of-thought is the
    **decompressed** version: each intermediate step is simple, even if the final
    answer is complex.
    """
)

col_comp, col_decomp = st.columns(2)
with col_comp:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#E74C3C;">Direct Answer (Compressed)</strong><br/><br/>'
        '<span style="font-family:monospace; font-size:1.2rem;">'
        'Q: What is 17 x 24?<br/>'
        'A: <strong>408</strong>'
        '</span><br/><br/>'
        'All reasoning is <em>implicit</em> -- packed into the model\'s forward pass. '
        'Requires enormous model capacity to get right.'
        '</div>',
        unsafe_allow_html=True,
    )
with col_decomp:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#2ECC71;">CoT Answer (Decompressed)</strong><br/><br/>'
        '<span style="font-family:monospace; font-size:1.2rem;">'
        'Q: What is 17 x 24?<br/>'
        'A: 17 x 24 = 17 x 20 + 17 x 4<br/>'
        '&nbsp;&nbsp; = 340 + 68 = <strong>408</strong>'
        '</span><br/><br/>'
        'Each step is <em>simple</em>. The complexity is spread across serial tokens. '
        'Trades compute for capacity.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="insight-box">'
    "<strong>The model learns a meta-compressor.</strong> It doesn't just learn to "
    "solve problems -- it learns <em>how to decompose</em> problems into simple steps. "
    "The CoT is a learned decompression algorithm: unpack a hard problem into a "
    "sequence of easy ones."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 2. GRPO vs PPO COMPARISON
# =====================================================================
st.markdown(
    '<p class="section-header">ALGORITHM COMPARISON</p>',
    unsafe_allow_html=True,
)
st.subheader("GRPO vs PPO: Efficient RL for Reasoning")

st.markdown(
    """
    **PPO** (Proximal Policy Optimization) is the workhorse of RLHF but requires
    four separate models in memory.  **GRPO** (Group Relative Policy Optimization),
    introduced with DeepSeek-R1, eliminates the value function and replaces the
    learned reward model with verifiable rewards -- cutting memory roughly in half.
    """
)

col_ppo_t, col_grpo_t = st.columns(2)
with col_ppo_t:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#4A90D9;">PPO (4 models)</strong><br/><br/>'
        "1. <strong>Policy model</strong> -- the model being trained<br/>"
        "2. <strong>Reference model</strong> -- frozen copy for KL penalty<br/>"
        "3. <strong>Reward model</strong> -- learned from human preferences<br/>"
        "4. <strong>Value function</strong> -- critic that estimates advantage<br/><br/>"
        '<span style="color:#E74C3C;">Memory: ~4x model parameters</span>'
        "</div>",
        unsafe_allow_html=True,
    )
with col_grpo_t:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#2ECC71;">GRPO (2 models)</strong><br/><br/>'
        "1. <strong>Policy model</strong> -- the model being trained<br/>"
        "2. <strong>Reference model</strong> -- frozen copy for KL penalty<br/>"
        '3. <span style="color:#95A5A6;">Verifiable reward (rule-based, free)</span><br/>'
        '4. <span style="color:#95A5A6;">Group baseline (computed from samples, free)</span><br/><br/>'
        '<span style="color:#2ECC71;">Memory: ~2x model parameters</span>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

st.markdown(
    '<div class="big-formula">'
    "GRPO advantage: &nbsp; A&#770;<sub>i</sub> = "
    "(r<sub>i</sub> - mean(r<sub>group</sub>)) / std(r<sub>group</sub>)"
    "<br/><br/>"
    "No value function needed -- the group of sampled completions IS the baseline."
    "</div>",
    unsafe_allow_html=True,
)

# -- Memory bar chart with slider
st.markdown("##### Memory Requirements by Model Size")

model_size_b = st.slider(
    "Model size (billions of parameters)",
    min_value=7,
    max_value=70,
    value=7,
    step=1,
    key="model_size_slider",
)

# Rough estimates: each model copy ~ 2 bytes/param (bf16)
bytes_per_param = 2  # bf16
gb_per_model = (model_size_b * 1e9 * bytes_per_param) / (1024**3)

ppo_models = 4
grpo_models = 2

# PPO also needs optimizer states for policy + value ~ 2x extra for those 2
# Simplified: PPO ~ 4 copies + optimizer overhead, GRPO ~ 2 copies + optimizer overhead
ppo_total_gb = gb_per_model * ppo_models * 1.3  # 1.3x for optimizer states
grpo_total_gb = gb_per_model * grpo_models * 1.3

ppo_param_total = model_size_b * ppo_models
grpo_param_total = model_size_b * grpo_models

fig_mem = go.Figure()
fig_mem.add_trace(go.Bar(
    x=["PPO", "GRPO"],
    y=[ppo_total_gb, grpo_total_gb],
    marker_color=[COLORS["blue"], COLORS["green"]],
    text=[f"{ppo_total_gb:.0f} GB<br>({ppo_param_total}B params)",
          f"{grpo_total_gb:.0f} GB<br>({grpo_param_total}B params)"],
    textposition="outside",
    textfont=dict(size=13),
))
fig_mem.update_layout(
    title=f"GPU Memory Estimate for {model_size_b}B Parameter Model (bf16)",
    yaxis_title="Memory (GB)",
    yaxis=dict(range=[0, max(ppo_total_gb, grpo_total_gb) * 1.25]),
    height=400,
    showlegend=False,
)

# Add component breakdown annotations
fig_mem.add_annotation(
    x="PPO", y=ppo_total_gb * 0.5,
    text="Policy + Ref + RM + Value<br>+ optimizer states",
    showarrow=False,
    font=dict(size=11, color=COLORS["white"]),
)
fig_mem.add_annotation(
    x="GRPO", y=grpo_total_gb * 0.5,
    text="Policy + Ref<br>+ optimizer states",
    showarrow=False,
    font=dict(size=11, color=COLORS["white"]),
)

st.plotly_chart(fig_mem, use_container_width=True)

savings_pct = (1 - grpo_total_gb / ppo_total_gb) * 100
st.markdown(
    f'<div class="insight-box">'
    f"<strong>GRPO saves ~{savings_pct:.0f}% GPU memory</strong> compared to PPO "
    f"at {model_size_b}B parameters. The key trick: instead of learning a value function "
    f"(expensive critic network), GRPO samples a <em>group</em> of completions and uses "
    f"their relative rewards as the baseline. Verifiable rewards (math correctness, code "
    f"execution) replace the learned reward model entirely."
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. EMERGENT BEHAVIORS SIMULATOR
# =====================================================================
st.markdown(
    '<p class="section-header">EMERGENT BEHAVIORS</p>',
    unsafe_allow_html=True,
)
st.subheader("Emergent Behaviors from Reward Maximization")

st.markdown(
    """
    The following behaviors were **NOT** explicitly trained.  They emerged
    spontaneously from reward maximization on verifiable tasks (math, code,
    logic puzzles).  The model discovered these strategies because they help
    maximize the probability of reaching the correct answer.
    """
)

behavior_tab = st.selectbox(
    "Select an emergent behavior to explore",
    [
        "Aha Moments (Backtracking & Reconsideration)",
        "Self-Verification (Checking Own Work)",
        "Progressive Lengthening (Longer Chains Over Training)",
        "Strategic Exploration (Multiple Approaches)",
    ],
    key="emergent_behavior_select",
)

if behavior_tab == "Aha Moments (Backtracking & Reconsideration)":
    st.markdown(
        """
        The model learns to **reconsider its approach** mid-reasoning when
        it detects an inconsistency.  This is analogous to a human having an
        "aha moment" -- realizing a mistake and pivoting.
        """
    )

    st.markdown(
        '<div class="concept-card" style="font-family: monospace; font-size: 0.92rem; '
        'line-height: 1.7;">'
        '<strong style="color:#4A90D9;">Simulated Reasoning Trace:</strong><br/><br/>'
        '<span style="color:#95A5A6;">Problem: Is 97 prime?</span><br/><br/>'
        '<span style="color:#ECF0F1;">Let me check if 97 is divisible by small primes.</span><br/>'
        '<span style="color:#ECF0F1;">97 / 2 = 48.5, not divisible.</span><br/>'
        '<span style="color:#ECF0F1;">97 / 3 = 32.33..., not divisible.</span><br/>'
        '<span style="color:#ECF0F1;">97 / 5 = 19.4, not divisible.</span><br/>'
        '<span style="color:#ECF0F1;">97 / 7 = 13.86..., not divisible.</span><br/>'
        '<span style="color:#F39C12;"><strong>Wait, I should also check if sqrt(97) &lt; 10, '
        'so I only need primes up to 9.</strong></span><br/>'
        '<span style="color:#2ECC71;">Since I have checked 2, 3, 5, 7 and all are less than '
        'sqrt(97) ~ 9.85, I am done.</span><br/>'
        '<span style="color:#2ECC71;"><strong>97 is prime.</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight-box">'
        '<strong>The "Wait" moment</strong> is the hallmark of an aha moment. '
        "The model learned that pausing to reconsider is rewarded because it leads to "
        "more correct final answers. Nobody taught it to say 'wait' -- this emerged "
        "from pure reward maximization."
        "</div>",
        unsafe_allow_html=True,
    )

elif behavior_tab == "Self-Verification (Checking Own Work)":
    st.markdown(
        """
        The model learns to **verify its own answer** before committing to it.
        This is a form of internal quality control that reduces error rates.
        """
    )

    st.markdown(
        '<div class="concept-card" style="font-family: monospace; font-size: 0.92rem; '
        'line-height: 1.7;">'
        '<strong style="color:#4A90D9;">Simulated Reasoning Trace:</strong><br/><br/>'
        '<span style="color:#95A5A6;">Problem: Solve 3x + 7 = 22</span><br/><br/>'
        '<span style="color:#ECF0F1;">Subtract 7 from both sides: 3x = 15</span><br/>'
        '<span style="color:#ECF0F1;">Divide by 3: x = 5</span><br/><br/>'
        '<span style="color:#9B59B6;"><strong>Let me verify: 3(5) + 7 = 15 + 7 = 22. '
        'Correct!</strong></span><br/><br/>'
        '<span style="color:#2ECC71;"><strong>x = 5</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="concept-card" style="font-family: monospace; font-size: 0.92rem; '
        'line-height: 1.7;">'
        '<strong style="color:#E74C3C;">Trace where self-verification catches an error:</strong>'
        '<br/><br/>'
        '<span style="color:#95A5A6;">Problem: What is 23 x 17?</span><br/><br/>'
        '<span style="color:#ECF0F1;">23 x 17 = 23 x 10 + 23 x 7</span><br/>'
        '<span style="color:#ECF0F1;">= 230 + 151</span><br/>'
        '<span style="color:#ECF0F1;">= 381</span><br/><br/>'
        '<span style="color:#9B59B6;"><strong>Let me check: 23 x 7 = 161, not 151.</strong></span><br/>'
        '<span style="color:#F39C12;"><strong>Correcting: 230 + 161 = 391</strong></span><br/><br/>'
        '<span style="color:#2ECC71;"><strong>23 x 17 = 391</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )

elif behavior_tab == "Progressive Lengthening (Longer Chains Over Training)":
    st.markdown(
        """
        As training progresses, the model's reasoning chains **get longer**.
        The model discovers that more thorough reasoning leads to higher reward,
        so it naturally allocates more "thinking tokens" to harder problems.
        """
    )

    # Generate simulated training data
    np.random.seed(42)
    training_steps = np.arange(0, 10001, 100)
    # Chain length grows with a log curve + noise
    base_length = 50 + 150 * np.log1p(training_steps / 500)
    noise = np.random.normal(0, 12, size=len(training_steps))
    chain_lengths = np.clip(base_length + noise, 20, None)

    # Also track accuracy improving
    accuracy = 0.35 + 0.55 * (1 - np.exp(-training_steps / 3000))
    acc_noise = np.random.normal(0, 0.02, size=len(training_steps))
    accuracy = np.clip(accuracy + acc_noise, 0.1, 0.98)

    fig_length = go.Figure()
    fig_length.add_trace(go.Scatter(
        x=training_steps,
        y=chain_lengths,
        mode="lines",
        name="Avg. Chain Length (tokens)",
        line=dict(color=COLORS["blue"], width=2.5),
        yaxis="y1",
    ))
    fig_length.add_trace(go.Scatter(
        x=training_steps,
        y=accuracy * 100,
        mode="lines",
        name="Accuracy (%)",
        line=dict(color=COLORS["green"], width=2.5, dash="dash"),
        yaxis="y2",
    ))
    fig_length.update_layout(
        title="Reasoning Chain Length & Accuracy Over Training",
        xaxis_title="Training Step",
        yaxis=dict(
            title="Avg. Chain Length (tokens)",
            titlefont=dict(color=COLORS["blue"]),
            tickfont=dict(color=COLORS["blue"]),
        ),
        yaxis2=dict(
            title="Accuracy (%)",
            titlefont=dict(color=COLORS["green"]),
            tickfont=dict(color=COLORS["green"]),
            overlaying="y",
            side="right",
            range=[0, 105],
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_length, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<strong>Correlation, not coincidence.</strong> Longer chains appear because "
        "the model is rewarded for correct answers, and more thorough reasoning "
        "leads to more correct answers. The model learns to 'budget' its thinking -- "
        "easy problems get short chains, hard problems get long ones."
        "</div>",
        unsafe_allow_html=True,
    )

elif behavior_tab == "Strategic Exploration (Multiple Approaches)":
    st.markdown(
        """
        When one approach fails or seems stuck, the model learns to **try a
        different strategy**.  This is analogous to a human saying "let me try
        another way."
        """
    )

    st.markdown(
        '<div class="concept-card" style="font-family: monospace; font-size: 0.92rem; '
        'line-height: 1.7;">'
        '<strong style="color:#4A90D9;">Simulated Reasoning Trace:</strong><br/><br/>'
        '<span style="color:#95A5A6;">Problem: Find all integers x such that '
        'x^2 - 5x + 6 = 0</span><br/><br/>'
        '<span style="color:#ECF0F1;"><strong>Approach 1: Quadratic formula</strong></span><br/>'
        '<span style="color:#ECF0F1;">x = (5 +/- sqrt(25 - 24)) / 2</span><br/>'
        '<span style="color:#ECF0F1;">x = (5 +/- 1) / 2</span><br/>'
        '<span style="color:#ECF0F1;">x = 3 or x = 2</span><br/><br/>'
        '<span style="color:#F39C12;"><strong>Let me also try factoring to confirm.</strong>'
        '</span><br/>'
        '<span style="color:#ECF0F1;"><strong>Approach 2: Factoring</strong></span><br/>'
        '<span style="color:#ECF0F1;">x^2 - 5x + 6 = (x - 2)(x - 3) = 0</span><br/>'
        '<span style="color:#ECF0F1;">x = 2 or x = 3</span><br/><br/>'
        '<span style="color:#2ECC71;"><strong>Both approaches agree: x = 2 and x = 3.'
        '</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight-box">'
        "<strong>Strategic exploration is a meta-skill.</strong> The model doesn't just "
        "learn one algorithm per problem type -- it learns to deploy multiple strategies "
        "and cross-check results. This emerges because cross-checking increases the "
        "probability of a correct final answer."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 4. PROCESS REWARD MODEL (PRM) DEMO
# =====================================================================
st.markdown(
    '<p class="section-header">PROCESS REWARD MODELS</p>',
    unsafe_allow_html=True,
)
st.subheader("Process Reward Model (PRM) Demo")

st.markdown(
    """
    Traditional reward models score only the **final answer** (Outcome Reward Model, ORM).
    Process Reward Models score **each reasoning step**, enabling fine-grained credit
    assignment.  This is critical for long chains where a single wrong step can derail
    the entire solution.
    """
)

col_orm, col_prm = st.columns(2)
with col_orm:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#E74C3C;">ORM (Outcome)</strong><br/>'
        "Scores only the final answer.<br/>"
        "Cheap but coarse -- doesn't know <em>where</em> reasoning went wrong."
        "</div>",
        unsafe_allow_html=True,
    )
with col_prm:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#2ECC71;">PRM (Process)</strong><br/>'
        "Scores every intermediate step.<br/>"
        "Expensive but precise -- pinpoints the <em>exact</em> step where errors occur."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("##### PRM Scoring Methods")
scoring_method = st.radio(
    "Choose a scoring method",
    ["Human Annotation", "Monte Carlo Estimation"],
    horizontal=True,
    key="prm_scoring_method",
)

if scoring_method == "Human Annotation":
    st.markdown(
        """
        Expert annotators label each reasoning step as **correct**, **neutral**, or
        **incorrect**.  This is the gold standard but extremely expensive -- a single
        multi-step math problem can take 10+ minutes to annotate.
        """
    )
else:
    st.markdown(
        """
        **Monte Carlo estimation**: From each intermediate step, generate K independent
        completions and check how many reach the correct final answer.  The step's
        score is the fraction of completions that succeed:

        **Score(step_i) = (# completions from step_i that reach correct answer) / K**

        Steps where the score drops sharply are where reasoning went wrong.
        """
    )

# -- Simulated multi-step reasoning with PRM scores
st.markdown("##### Example: Multi-Step Math Problem with Step-Level Scores")

problem_choice = st.selectbox(
    "Select a problem scenario",
    [
        "Correct reasoning (all steps valid)",
        "Error at Step 4 (algebraic mistake)",
        "Error at Step 2 (wrong approach, early derailment)",
    ],
    key="prm_problem_choice",
)

if problem_choice == "Correct reasoning (all steps valid)":
    steps = [
        "Read problem: Solve 2x^2 + 3x - 5 = 0",
        "Identify: quadratic equation, use quadratic formula",
        "Compute discriminant: b^2 - 4ac = 9 + 40 = 49",
        "Take sqrt: sqrt(49) = 7",
        "Apply formula: x = (-3 +/- 7) / 4",
        "Solution: x = 1 or x = -5/2",
    ]
    if scoring_method == "Human Annotation":
        scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        scores = [0.95, 0.93, 0.91, 0.90, 0.92, 0.94]
    error_step = None

elif problem_choice == "Error at Step 4 (algebraic mistake)":
    steps = [
        "Read problem: Solve 2x^2 + 3x - 5 = 0",
        "Identify: quadratic equation, use quadratic formula",
        "Compute discriminant: b^2 - 4ac = 9 + 40 = 49",
        "Take sqrt: sqrt(49) = 8  [ERROR: should be 7]",
        "Apply formula: x = (-3 +/- 8) / 4",
        "Solution: x = 5/4 or x = -11/4  [WRONG]",
    ]
    if scoring_method == "Human Annotation":
        scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    else:
        scores = [0.94, 0.91, 0.88, 0.12, 0.08, 0.05]
    error_step = 3

elif problem_choice == "Error at Step 2 (wrong approach, early derailment)":
    steps = [
        "Read problem: Solve 2x^2 + 3x - 5 = 0",
        "Try factoring: (2x - 1)(x + 5) = 0  [ERROR: wrong factors]",
        "From first factor: x = 1/2",
        "From second factor: x = -5",
        "Check: 2(1/4) + 3(1/2) - 5 = 0.5 + 1.5 - 5 = -3 != 0",
        "Something is wrong but commit anyway: x = 1/2, x = -5  [WRONG]",
    ]
    if scoring_method == "Human Annotation":
        scores = [1.0, 0.0, 0.0, 0.0, 0.5, 0.0]
    else:
        scores = [0.93, 0.15, 0.10, 0.08, 0.30, 0.04]
    error_step = 1

# Build the PRM visualization
step_labels = [f"Step {i+1}" for i in range(len(steps))]

bar_colors_prm = []
for i, s in enumerate(scores):
    if s >= 0.7:
        bar_colors_prm.append(COLORS["green"])
    elif s >= 0.3:
        bar_colors_prm.append(COLORS["orange"])
    else:
        bar_colors_prm.append(COLORS["red"])

fig_prm = go.Figure()
fig_prm.add_trace(go.Bar(
    x=step_labels,
    y=scores,
    marker_color=bar_colors_prm,
    text=[f"{s:.2f}" for s in scores],
    textposition="outside",
    textfont=dict(size=12),
    hovertext=steps,
    hoverinfo="text+y",
))
fig_prm.update_layout(
    title="Process Reward: Step-Level Scores",
    xaxis_title="Reasoning Step",
    yaxis_title="Score" if scoring_method == "Human Annotation" else "P(correct | step)",
    yaxis=dict(range=[0, 1.15]),
    height=380,
    showlegend=False,
)

if error_step is not None:
    fig_prm.add_annotation(
        x=step_labels[error_step],
        y=scores[error_step] + 0.08,
        text="Error here!",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["red"],
        font=dict(color=COLORS["red"], size=13, family="Inter"),
        ax=0, ay=-35,
    )

st.plotly_chart(fig_prm, use_container_width=True)

# Show the reasoning steps as a table
st.markdown("##### Reasoning Trace")
for i, (step, score) in enumerate(zip(steps, scores)):
    if score >= 0.7:
        color = COLORS["green"]
        icon = "&#10003;"
    elif score >= 0.3:
        color = COLORS["orange"]
        icon = "&#9888;"
    else:
        color = COLORS["red"]
        icon = "&#10007;"

    st.markdown(
        f'<div style="background:{COLORS["card"]}; border-left:4px solid {color}; '
        f'padding:8px 14px; margin:4px 0; border-radius:6px; font-size:0.95rem;">'
        f'<span style="color:{color}; font-weight:bold;">{icon} Step {i+1}</span> '
        f'<span style="color:{COLORS["gray"]};">[score: {score:.2f}]</span><br/>'
        f'<span style="color:{COLORS["white"]};">{step}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

if scoring_method == "Monte Carlo Estimation":
    st.markdown("")
    st.markdown(
        '<div class="insight-box">'
        "<strong>How Monte Carlo PRM works:</strong> At each step, we branch off K "
        "independent completions (e.g., K=64). If step 3 has score 0.88, it means "
        "that 88% of completions starting from step 3 eventually reach the correct "
        "answer. When the score drops sharply (e.g., from 0.88 to 0.12), we have "
        "found the critical error step -- <em>without any human labels</em>."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 5. COT AS DECOMPRESSION -- COMPUTE vs CAPACITY TRADEOFF
# =====================================================================
st.markdown(
    '<p class="section-header">THEORY</p>',
    unsafe_allow_html=True,
)
st.subheader("CoT as Decompression: Compute vs Capacity Tradeoff")

st.markdown(
    """
    There is a fundamental tradeoff between **model capacity** (parameters) and
    **inference compute** (thinking tokens).  A larger model can solve problems
    in fewer steps; a smaller model needs more chain-of-thought tokens to reach
    the same answer.  CoT is literally **trading compute for capacity**.
    """
)

# Interactive: show how direct answer requires high capacity, CoT reduces it
st.markdown("##### The Compression Frame")

col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    problem_difficulty = st.slider(
        "Problem difficulty (Kolmogorov complexity K)",
        min_value=1.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="prob_difficulty",
    )
with col_slider2:
    cot_steps = st.slider(
        "Number of CoT reasoning steps",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        key="cot_steps_slider",
    )

# Model capacity needed decreases with more CoT steps
# Direct answer: need capacity >= K
# With CoT: each step reduces remaining complexity
# capacity_needed = K / (1 + alpha * steps)
alpha = 0.3
capacity_needed_direct = problem_difficulty
capacity_needed_cot = problem_difficulty / (1 + alpha * cot_steps)
inference_cost = 1.0 + cot_steps * 0.5  # relative cost

# Generate curve data
steps_range = np.arange(0, 21)
capacity_curve = problem_difficulty / (1 + alpha * steps_range)
compute_curve = 1.0 + steps_range * 0.5

fig_tradeoff = go.Figure()
fig_tradeoff.add_trace(go.Scatter(
    x=steps_range,
    y=capacity_curve,
    mode="lines+markers",
    name="Required Model Capacity",
    line=dict(color=COLORS["blue"], width=3),
    marker=dict(size=6),
))
fig_tradeoff.add_trace(go.Scatter(
    x=steps_range,
    y=compute_curve,
    mode="lines+markers",
    name="Inference Compute (relative)",
    line=dict(color=COLORS["orange"], width=3, dash="dash"),
    marker=dict(size=6),
))

# Highlight current selection
fig_tradeoff.add_trace(go.Scatter(
    x=[cot_steps],
    y=[capacity_needed_cot],
    mode="markers",
    marker=dict(size=16, color=COLORS["blue"], symbol="star", line=dict(width=2, color="white")),
    name=f"Current: {cot_steps} steps",
    showlegend=True,
))
fig_tradeoff.add_trace(go.Scatter(
    x=[cot_steps],
    y=[inference_cost],
    mode="markers",
    marker=dict(size=16, color=COLORS["orange"], symbol="star", line=dict(width=2, color="white")),
    showlegend=False,
))

fig_tradeoff.update_layout(
    title=f"Capacity vs Compute Tradeoff (Problem K = {problem_difficulty:.1f})",
    xaxis_title="Number of CoT Steps",
    yaxis_title="Relative Scale",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_tradeoff, use_container_width=True)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric(
    "Capacity (direct)",
    f"{capacity_needed_direct:.1f}",
    help="Model capacity needed for direct answer (no CoT)",
)
m2.metric(
    f"Capacity ({cot_steps} CoT steps)",
    f"{capacity_needed_cot:.1f}",
    delta=f"-{(1 - capacity_needed_cot / capacity_needed_direct) * 100:.0f}%",
    delta_color="normal",
)
m3.metric(
    "Inference cost",
    f"{inference_cost:.1f}x",
    delta=f"+{(inference_cost - 1) * 100:.0f}%",
    delta_color="inverse",
)

# Visual comparison: direct vs CoT
st.markdown("")
col_direct, col_cot_vis = st.columns(2)
with col_direct:
    st.markdown(
        '<div class="concept-card" style="text-align:center;">'
        f'<strong style="color:{COLORS["red"]};">Direct Answer</strong><br/><br/>'
        f'<span style="font-size:2rem; font-weight:bold;">"{42 if problem_difficulty <= 5 else "???"}"</span>'
        '<br/><br/>'
        f'<span style="color:{COLORS["gray"]};">All K={problem_difficulty:.1f} bits of complexity<br/>'
        'compressed into one forward pass</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with col_cot_vis:
    if cot_steps == 0:
        cot_display = "No CoT steps -- same as direct answer"
    else:
        bits_per_step = problem_difficulty / cot_steps
        step_strs = [f"Step {i+1}: ~{bits_per_step:.1f} bits" for i in range(min(cot_steps, 6))]
        if cot_steps > 6:
            step_strs.append(f"... ({cot_steps - 6} more steps)")
        cot_display = "<br/>".join(step_strs)

    st.markdown(
        '<div class="concept-card" style="text-align:center;">'
        f'<strong style="color:{COLORS["green"]};">CoT Answer</strong><br/><br/>'
        f'<span style="font-family:monospace; font-size:0.9rem; text-align:left; '
        f'display:inline-block;">{cot_display}</span>'
        '<br/><br/>'
        f'<span style="color:{COLORS["gray"]};">K={problem_difficulty:.1f} bits spread across '
        f'{max(cot_steps, 1)} steps<br/>'
        f'~{problem_difficulty / max(cot_steps, 1):.1f} bits per step</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="insight-box">'
    "<strong>The decompression principle:</strong> A direct answer packs all "
    "reasoning into a single forward pass (high capacity required). CoT "
    "decompresses this into a serial chain where each step is simple. "
    "This is why smaller models with CoT can outperform larger models "
    "without it -- they are using inference compute as a substitute for "
    "parameters."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 6. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.1rem; padding:24px 28px;">'
    "<strong>Key Insight</strong><br/><br/>"
    "These behaviors -- aha moments, self-verification, progressive lengthening, "
    "strategic exploration -- were <strong>NOT</strong> explicitly trained. They "
    "<strong>emerged from reward maximization</strong> on verifiable tasks "
    "(math, code, logic).<br/><br/>"
    "A good reasoning step <em>reduces the remaining Kolmogorov complexity</em> of "
    "the problem. The model learns a policy that decomposes hard problems into simpler "
    "sub-problems, because that is the strategy that maximizes the probability of "
    "eventually reaching a correct (and therefore rewarded) answer.<br/><br/>"
    "This is the deep connection: <strong>reasoning is learned compression at "
    "inference time</strong>."
    "</div>",
    unsafe_allow_html=True,
)
