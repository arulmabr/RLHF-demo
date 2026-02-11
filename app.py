import streamlit as st

st.set_page_config(
    page_title="Post-Training Foundations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from style import inject_custom_css
inject_custom_css()

st.markdown('<p class="section-header">UC BERKELEY CDSS 94 &mdash; POSTTRAINING.AI</p>', unsafe_allow_html=True)
st.title("Post-Training Foundations")
st.markdown("#### Interactive Platform &mdash; Lecture 3")

st.markdown("---")

st.markdown("""
### Navigate the Sections

Use the **sidebar** to explore each post-training technique interactively.
Each page contains brief explanations, live demos with sliders, and key insights
drawn from the 12 sections of Lecture 3.

""")

cols = st.columns(3)

sections = [
    ("I. The Big Idea", "Prediction = Compression. Play with entropy, zlib, and predictability.", "01"),
    ("II. SFT", "Simulate supervised fine-tuning: distribution shifts and failure modes.", "02"),
    ("III. KL Divergence", "Interactive KL calculator, forward vs reverse KL, KL budget.", "03"),
    ("IV. Distillation", "Temperature scaling, soft labels, CoT distillation.", "04"),
    ("V. Reward Modeling", "Be the reward model: Bradley-Terry, bias sliders, Goodhart.", "05"),
    ("VI. RLHF / PPO", "Boltzmann tilt, PPO clipping, the full training loop.", "06"),
    ("VII. DPO", "Implicit reward, gradient weighting, beta sensitivity.", "07"),
    ("VIII. Constitutional AI", "Critique-revise loop, over-refusal tradeoff.", "08"),
    ("IX. Best-of-N", "Rejection sampling visualization, KL cost, iterated BoN.", "09"),
    ("X. Verifiable Rewards", "Sparse binary reward, emergence threshold.", "10"),
    ("XI. Reasoning", "GRPO vs PPO, emergent behaviors, process reward models.", "11"),
    ("XII. Agentic & Eval", "Multi-turn challenges, the eval crisis.", "12"),
    ("XIII. Frontier", "Compression frame summary, method comparison.", "13"),
]

for i, (title, desc, _) in enumerate(sections):
    with cols[i % 3]:
        st.markdown(f"""
<div class="concept-card">
<strong>{title}</strong><br/>
<span style="color:#95A5A6;font-size:0.9rem;">{desc}</span>
</div>
""", unsafe_allow_html=True)
