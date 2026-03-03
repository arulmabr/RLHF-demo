import streamlit as st

st.set_page_config(
    page_title="RLHF Interactive Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from style import inject_custom_css, COLORS
inject_custom_css()

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
<div>
<p class="section-header" style="margin:0;">CDSS 94 &mdash; BUILDING THOUGHTFUL AI SYSTEMS</p>
<p style="color:{COLORS['gray']}; font-size:0.8rem; margin:0;">
UC Berkeley &bull; Spring 2026 &bull; Instructors: Karina Nguyen &amp; Kevin Miao &bull;
<a href="https://www.posttraining.ai/syllabus" style="color:{COLORS['blue']};">posttraining.ai</a>
</p>
</div>
</div>
""", unsafe_allow_html=True)
st.title("Post-Training Interactive Labs")
st.markdown("#### Hands-on demos for understanding alignment, evaluation, and model behavior")

st.markdown("---")

# ── Pipeline diagram ──
st.markdown("### The Post-Training Pipeline")

cols = st.columns(5)
steps = [
    ("1. Pretrain", "Train on internet text. Learns language, not behavior."),
    ("2. SFT", "Fine-tune on curated examples. Learns to follow instructions."),
    ("3. Human Feedback", "Humans compare outputs and pick the better one."),
    ("4. Reward Model", "A model that scores outputs based on human preferences."),
    ("5. RLHF", "Optimize the LLM to get high reward while staying close to SFT."),
]

for col, (title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
<div class="concept-card" style="min-height:140px;">
<strong style="color:{COLORS['blue']}">{title}</strong><br/><br/>
<span style="color:#95A5A6;font-size:0.85rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
### Lecture 3 — Post-Training Foundations

<div style="color:{COLORS['gray']}; font-size:0.85rem; margin-bottom:12px;">
RLHF pipeline, reward modeling, and alignment tradeoffs
</div>
""", unsafe_allow_html=True)

l3_cols = st.columns(5)
l3_pages = [
    ("Interactive RLHF", "You are the annotator. Pick preferences, train a reward model, optimize a policy."),
    ("Key Concepts", "KL divergence, reward hacking, DPO — the math behind alignment."),
    ("Spot the Hack", "Quiz: identify sycophancy, length gaming, format exploits, and more."),
    ("Best-of-N", "Inference-time scaling playground with RM noise and diminishing returns."),
    ("Disagreement", "5 annotators with different values — why human feedback is noisy."),
]
for col, (title, desc) in zip(l3_cols, l3_pages):
    with col:
        st.markdown(f"""
<div class="concept-card" style="min-height:120px;">
<strong style="color:{COLORS['blue']}">{title}</strong><br/><br/>
<span style="color:#95A5A6;font-size:0.82rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
### Lecture 5 — Evals, Debugging & Alignment

<div style="color:{COLORS['gray']}; font-size:0.85rem; margin-bottom:12px;">
Benchmarks, model behavior evaluation, calibration, and why evals break
</div>
""", unsafe_allow_html=True)

l5_cols = st.columns(4)
l5_pages = [
    ("Where's the Line?", "Classify prompts as safe or harmful — then see how cautious vs. permissive models compare to you."),
    ("Judge the Judge", "Mini Chatbot Arena: expose length, format, and position biases in LLM-as-a-judge evaluation."),
    ("Benchmark Decay", "Watch benchmarks saturate, get contaminated, and break from formatting changes."),
    ("How Confident?", "Explore calibration — the gap between a model saying '95% sure' and actually being right."),
]
for col, (title, desc) in zip(l5_cols, l5_pages):
    with col:
        st.markdown(f"""
<div class="concept-card" style="min-height:120px;">
<strong style="color:{COLORS['orange']}">{title}</strong><br/><br/>
<span style="color:#95A5A6;font-size:0.82rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)
