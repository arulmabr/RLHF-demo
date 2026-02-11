import streamlit as st

st.set_page_config(
    page_title="RLHF Interactive Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from style import inject_custom_css, COLORS
inject_custom_css()

st.markdown('<p class="section-header">UC BERKELEY CDSS 94 &mdash; POSTTRAINING.AI</p>', unsafe_allow_html=True)
st.title("How RLHF Actually Works")
st.markdown("#### An interactive walkthrough of post-training alignment")

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

st.markdown("""
### Try It Yourself

Use the **sidebar** to navigate:

- **Interactive RLHF** — *You* are the human annotator. Pick preferred responses,
  watch a reward model learn your preferences, then see how RLHF optimizes the policy.
- **Key Concepts** — KL divergence, reward hacking, DPO, and why this is hard.
""")
