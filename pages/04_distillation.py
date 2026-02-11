import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from style import inject_custom_css, COLORS, softmax, kl_divergence

inject_custom_css()

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">SECTION IV &mdash; DISTILLATION</p>',
    unsafe_allow_html=True,
)
st.title("Distillation: Compressing Knowledge")

st.markdown("""
<div class="big-formula">
L = &alpha; &middot; CE(y<sub>hard</sub>, student) + (1 &minus; &alpha;) &middot; &tau;&sup2; &middot; KL(teacher<sub>soft</sub> || student<sub>soft</sub>)
</div>
""", unsafe_allow_html=True)

# ── Explanation ──────────────────────────────────────────────────────────────
st.markdown("---")

st.markdown("""
### Why Distillation?

Large "teacher" models are expensive to serve. **Distillation** trains a smaller
"student" model to reproduce the teacher's behavior -- not by copying the hard
one-hot labels, but by matching the teacher's **soft probability distribution**
over the entire vocabulary.

Why soft targets? Because the teacher's probability spread encodes valuable
structure. When a teacher assigns 0.6 to "happy" and 0.3 to "glad", that
similarity signal is lost if we only train on the argmax label "happy".

**Temperature** is the key mechanism: dividing logits by a temperature
parameter $\\tau > 1$ before the softmax *flattens* the distribution, revealing
the teacher's **dark knowledge** -- the relative ranking of tokens the teacher
considered plausible but ultimately did not choose.
""")

st.markdown("---")

# ── Section 1: Temperature Scaling Demo ──────────────────────────────────────
st.markdown("### 1. Temperature Scaling Demo")

st.markdown("""
Below are a teacher model's raw logits for six candidate tokens.
Adjust the temperature to see how the softmax output changes:
- **Low temperature** ($\\tau \\to 0$): distribution collapses to a hard argmax.
- **High temperature** ($\\tau \\to \\infty$): distribution flattens to uniform,
  revealing the relative ranking among *all* candidates.
""")

# Teacher logits (fixed)
TOKEN_LABELS = ["happy", "glad", "joyful", "sad", "angry", "ok"]
TEACHER_LOGITS = np.array([5.0, 3.8, 3.2, 0.5, -0.3, 1.0])

temperature = st.slider(
    "Temperature (\u03c4)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    key="temp_scaling",
)

hard_probs = softmax(TEACHER_LOGITS, temperature=0.01)  # near-argmax
soft_probs = softmax(TEACHER_LOGITS, temperature=temperature)

col_hard, col_soft = st.columns(2)

with col_hard:
    fig_hard = go.Figure()
    fig_hard.add_trace(go.Bar(
        x=TOKEN_LABELS,
        y=hard_probs,
        marker_color=COLORS["gray"],
        text=[f"{v:.3f}" for v in hard_probs],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_hard.update_layout(
        title="Hard Labels (\u03c4 \u2192 0)",
        yaxis=dict(title="Probability", range=[0, 1.08]),
        xaxis=dict(title="Token"),
        height=400,
    )
    st.plotly_chart(fig_hard, use_container_width=True)

with col_soft:
    # Color bars by magnitude so the "dark knowledge" tokens stand out
    bar_colors = [
        COLORS["blue"] if p > 0.05 else COLORS["purple"]
        for p in soft_probs
    ]
    fig_soft = go.Figure()
    fig_soft.add_trace(go.Bar(
        x=TOKEN_LABELS,
        y=soft_probs,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in soft_probs],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_soft.update_layout(
        title=f"Soft Labels (\u03c4 = {temperature:.1f})",
        yaxis=dict(title="Probability", range=[0, max(soft_probs) * 1.25 + 0.02]),
        xaxis=dict(title="Token"),
        height=400,
    )
    st.plotly_chart(fig_soft, use_container_width=True)

# Entropy readout
from style import entropy as _entropy

ent_hard = _entropy(hard_probs)
ent_soft = _entropy(soft_probs)
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Hard-label entropy", f"{ent_hard:.3f} bits")
col_m2.metric("Soft-label entropy", f"{ent_soft:.3f} bits")
col_m3.metric("Max possible entropy", f"{np.log2(len(TOKEN_LABELS)):.3f} bits")

st.markdown("""
<div class="insight-box">
<strong>Dark knowledge:</strong> At &tau; = 1 the distribution is already
somewhat peaked. Raising &tau; to 3 &ndash; 5 spreads mass onto the lower-ranked
tokens, exposing the teacher's internal similarity structure that the student
can learn from.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 2: Distillation Loss Playground ──────────────────────────────────
st.markdown("### 2. Distillation Loss Playground")

st.markdown("""
The distillation objective is a **weighted mix** of two losses:

| Component | What it captures |
|-----------|-----------------|
| $\\text{CE}(y_{\\text{hard}},\\, \\hat{y}_{\\text{student}})$ | Alignment with ground-truth labels |
| $\\tau^{2}\\; \\text{KL}(p_{\\text{teacher}}^{\\tau} \\| p_{\\text{student}}^{\\tau})$ | Matching the teacher's full soft distribution |

The parameter $\\alpha$ controls the blend; $\\tau$ controls softness.
""")

col_s1, col_s2 = st.columns(2)
with col_s1:
    alpha = st.slider(
        "Alpha (\u03b1) -- weight on hard-label CE",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="alpha_slider",
    )
with col_s2:
    tau = st.slider(
        "Temperature (\u03c4) for soft labels",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="tau_slider",
    )

# Simulate a student's logits (slightly worse than teacher)
STUDENT_LOGITS = np.array([4.0, 3.0, 2.5, 1.2, 0.3, 1.5])

# Hard label: one-hot on argmax of teacher
hard_label = np.zeros(len(TEACHER_LOGITS))
hard_label[np.argmax(TEACHER_LOGITS)] = 1.0

student_probs_hard = softmax(STUDENT_LOGITS, temperature=1.0)
teacher_soft = softmax(TEACHER_LOGITS, temperature=tau)
student_soft = softmax(STUDENT_LOGITS, temperature=tau)

# CE with hard label
ce_hard = -np.sum(hard_label * np.log(np.clip(student_probs_hard, 1e-10, 1.0)))

# KL soft (teacher || student)
kl_soft = kl_divergence(teacher_soft, student_soft)
kl_term = (tau ** 2) * kl_soft

total_loss = alpha * ce_hard + (1 - alpha) * kl_term

# Bar chart of loss components
fig_loss = go.Figure()
components = ["CE(hard)", f"\u03c4\u00b2 \u00b7 KL(soft)", "Total Loss"]
values = [alpha * ce_hard, (1 - alpha) * kl_term, total_loss]
colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

fig_loss.add_trace(go.Bar(
    x=components,
    y=values,
    marker_color=colors,
    text=[f"{v:.4f}" for v in values],
    textposition="outside",
    textfont=dict(size=13),
))
fig_loss.update_layout(
    title=f"Distillation Loss Breakdown  (\u03b1={alpha:.2f}, \u03c4={tau:.1f})",
    yaxis=dict(title="Loss value", range=[0, max(values) * 1.35 + 0.01]),
    height=400,
)
st.plotly_chart(fig_loss, use_container_width=True)

col_l1, col_l2, col_l3 = st.columns(3)
col_l1.metric("\u03b1 \u00b7 CE(hard)", f"{alpha * ce_hard:.4f}")
col_l2.metric("(1-\u03b1) \u00b7 \u03c4\u00b2 \u00b7 KL(soft)", f"{(1 - alpha) * kl_term:.4f}")
col_l3.metric("Total Loss", f"{total_loss:.4f}")

# ── What transfers vs what doesn't ──────────────────────────────────────────
st.markdown("#### What Distillation Transfers (and What It Doesn't)")

col_t, col_nt = st.columns(2)

with col_t:
    st.markdown("""
<div class="concept-card" style="border-left: 4px solid #2ECC71;">
<strong style="color:#2ECC71;">Transfers Well (Low KL budget)</strong><br/><br/>
<ul style="margin:0; padding-left:18px; color:#ECF0F1;">
<li>Style, tone, and formatting conventions</li>
<li>Simple behavioral rules and guardrails</li>
<li>Common factual recall</li>
<li>Surface-level instruction following</li>
<li>Language fluency and coherence</li>
</ul>
</div>
""", unsafe_allow_html=True)

with col_nt:
    st.markdown("""
<div class="concept-card" style="border-left: 4px solid #E74C3C;">
<strong style="color:#E74C3C;">Does NOT Transfer (High KL budget)</strong><br/><br/>
<ul style="margin:0; padding-left:18px; color:#ECF0F1;">
<li>Multi-step reasoning and planning</li>
<li>Rare edge-case handling</li>
<li>Novel problem decomposition</li>
<li>Calibrated uncertainty (knows what it doesn't know)</li>
<li>Long-horizon agentic behaviour</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 3: CoT Distillation Demo ─────────────────────────────────────────
st.markdown("### 3. Chain-of-Thought (CoT) Distillation")

st.markdown("""
Standard distillation transfers the teacher's **final answer distribution**.
**CoT distillation** goes further: it distills the teacher's *reasoning traces*
so the student learns **how to think**, not just what to output.
""")

# Pipeline visualisation
st.markdown("#### The 3-Step Pipeline")

step_cols = st.columns(3)

with step_cols[0]:
    st.markdown("""
<div class="concept-card" style="border-left: 4px solid #4A90D9; min-height: 200px;">
<strong style="color:#4A90D9;">Step 1 &mdash; Generate</strong><br/><br/>
<span style="color:#ECF0F1;">
The <strong>large teacher</strong> model solves problems while producing
explicit chain-of-thought reasoning traces.<br/><br/>
<em style="color:#95A5A6;">Example: "First I identify the
variables, then set up the equation, then solve for x..."</em>
</span>
</div>
""", unsafe_allow_html=True)

with step_cols[1]:
    st.markdown("""
<div class="concept-card" style="border-left: 4px solid #F39C12; min-height: 200px;">
<strong style="color:#F39C12;">Step 2 &mdash; Distill via SFT</strong><br/><br/>
<span style="color:#ECF0F1;">
Fine-tune the <strong>smaller student</strong> on the teacher's
(input, CoT reasoning, answer) triples using standard SFT.<br/><br/>
<em style="color:#95A5A6;">The student learns to produce
the reasoning steps, not just mimic the final token distribution.</em>
</span>
</div>
""", unsafe_allow_html=True)

with step_cols[2]:
    st.markdown("""
<div class="concept-card" style="border-left: 4px solid #2ECC71; min-height: 200px;">
<strong style="color:#2ECC71;">Step 3 &mdash; Evaluate</strong><br/><br/>
<span style="color:#ECF0F1;">
The student now <strong>generates its own reasoning</strong> at inference
time, arriving at correct answers through learned thought patterns.<br/><br/>
<em style="color:#95A5A6;">Accuracy approaches the teacher
because the student internalised the process.</em>
</span>
</div>
""", unsafe_allow_html=True)

# ── Simulated accuracy comparison ────────────────────────────────────────────
st.markdown("#### Simulated Accuracy Comparison")

st.markdown("""
Below we show simulated benchmark accuracy across four difficulty tiers.
The **CoT-distilled student** significantly outperforms the answer-only student,
especially on harder problems where reasoning matters most.
""")

difficulty_levels = ["Easy", "Medium", "Hard", "Very Hard"]

# Simulated accuracy data (percentages)
teacher_acc = np.array([97, 91, 82, 68])
student_answer_only = np.array([93, 78, 55, 30])
student_cot = np.array([95, 87, 74, 58])

fig_acc = go.Figure()

fig_acc.add_trace(go.Bar(
    name="Teacher (large)",
    x=difficulty_levels,
    y=teacher_acc,
    marker_color=COLORS["blue"],
    text=[f"{v}%" for v in teacher_acc],
    textposition="outside",
    textfont=dict(size=12),
))
fig_acc.add_trace(go.Bar(
    name="Student (answer-only distill)",
    x=difficulty_levels,
    y=student_answer_only,
    marker_color=COLORS["red"],
    text=[f"{v}%" for v in student_answer_only],
    textposition="outside",
    textfont=dict(size=12),
))
fig_acc.add_trace(go.Bar(
    name="Student (CoT distill)",
    x=difficulty_levels,
    y=student_cot,
    marker_color=COLORS["green"],
    text=[f"{v}%" for v in student_cot],
    textposition="outside",
    textfont=dict(size=12),
))

fig_acc.update_layout(
    title="Accuracy by Problem Difficulty",
    xaxis=dict(title="Difficulty Tier"),
    yaxis=dict(title="Accuracy (%)", range=[0, 110]),
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    height=480,
)

st.plotly_chart(fig_acc, use_container_width=True)

# Gap analysis
gap_answer = teacher_acc - student_answer_only
gap_cot = teacher_acc - student_cot

fig_gap = go.Figure()
fig_gap.add_trace(go.Scatter(
    x=difficulty_levels,
    y=gap_answer,
    mode="lines+markers+text",
    name="Answer-only gap",
    line=dict(color=COLORS["red"], width=3),
    marker=dict(size=10),
    text=[f"{v}pp" for v in gap_answer],
    textposition="top center",
    textfont=dict(size=12),
))
fig_gap.add_trace(go.Scatter(
    x=difficulty_levels,
    y=gap_cot,
    mode="lines+markers+text",
    name="CoT distill gap",
    line=dict(color=COLORS["green"], width=3),
    marker=dict(size=10),
    text=[f"{v}pp" for v in gap_cot],
    textposition="top center",
    textfont=dict(size=12),
))
fig_gap.update_layout(
    title="Accuracy Gap vs Teacher (lower is better)",
    xaxis=dict(title="Difficulty Tier"),
    yaxis=dict(title="Gap (percentage points)", range=[0, max(gap_answer) * 1.4]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    height=400,
)
st.plotly_chart(fig_gap, use_container_width=True)

st.markdown("""
<div class="insight-box">
<strong>Pattern:</strong> The answer-only student degrades sharply on hard
problems (&minus;38pp on "Very Hard") because it only learned the surface
mapping. The CoT student retains more of the teacher's capability (&minus;10pp)
because it learned the <em>reasoning process</em> itself.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 4: Key Insight ───────────────────────────────────────────────────
st.markdown("### Key Insight")

st.markdown("""
<div class="insight-box" style="border-left-color: #E74C3C; font-size: 1.05rem; padding: 20px 24px;">
<strong style="color:#E74C3C;">The Imitation Gap</strong><br/><br/>
Distilled models <em>sound</em> good but fail on hard problems. They learn the
<strong>style of intelligence before the substance.</strong><br/><br/>
A distilled model can perfectly mimic the teacher's tone, formatting, and
surface-level behaviour &mdash; yet collapse on multi-step reasoning, novel
edge cases, or calibration under uncertainty. This is because soft-label
matching rewards distributional similarity, not functional competence.<br/><br/>
CoT distillation narrows this gap by forcing the student to reproduce the
teacher's <em>intermediate reasoning</em>, but even then a capability ceiling
remains: the student's smaller capacity fundamentally limits what reasoning
chains it can reliably execute.
</div>
""", unsafe_allow_html=True)
