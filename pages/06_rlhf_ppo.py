"""
Page 06 -- RLHF with PPO
Covers the RLHF objective, Boltzmann tilt, PPO clipping, a simulated
training loop, and memory / hyperparameter challenges.
"""

from style import inject_custom_css, COLORS, softmax, kl_divergence, sigmoid
import streamlit as st
import numpy as np
import plotly.graph_objects as go

inject_custom_css()

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">SECTION VI</p>', unsafe_allow_html=True)
st.title("RLHF with PPO")
st.markdown(
    "Reinforcement Learning from Human Feedback turns a reward model's "
    "signal into a better language model -- but the optimisation is a "
    "balancing act between **reward maximisation** and **staying close to "
    "the reference policy**."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. THE RLHF OBJECTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("1  The RLHF Objective")

st.markdown(r"""
<div class="big-formula">
max<sub>&pi;<sub>&theta;</sub></sub> &nbsp;
E<sub>x ~ D, y ~ &pi;<sub>&theta;</sub>(y|x)</sub>
[ r(x, y) ] &minus; &beta; &middot; KL( &pi;<sub>&theta;</sub> || &pi;<sub>ref</sub> )
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
<div class="concept-card">
<strong style="color:{c};">Force 1 -- Reward Maximisation</strong><br/>
Push the policy toward completions the reward model scores highly.
Left unchecked this leads to <em>reward hacking</em>: the model discovers
adversarial outputs that exploit RM weaknesses.
</div>
""".format(c=COLORS["green"]), unsafe_allow_html=True)
with col_r:
    st.markdown("""
<div class="concept-card">
<strong style="color:{c};">Force 2 -- KL Penalty</strong><br/>
Pull the policy back toward the reference (usually the SFT model).
This regularises training and prevents mode collapse, but too strong
a pull means you never learn anything new.
</div>
""".format(c=COLORS["red"]), unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
<strong>The tension:</strong> Reward says "go far", KL says "stay home".
&beta; sets the exchange rate.  The closed-form optimum is
&pi;*(y|x) &prop; &pi;<sub>ref</sub>(y|x) &middot; exp(r(x,y) / &beta;).
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BOLTZMANN TILT VISUALISER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("2  Boltzmann Tilt Visualiser")
st.markdown(
    "The optimal RLHF policy tilts the reference distribution toward "
    "high-reward responses.  Beta controls how aggressively."
)

RESPONSE_TYPES = [
    "Helpful\ncorrect",
    "Helpful\nverbose",
    "Concise\ncorrect",
    "Sycophantic",
    "Refusal\n(safe)",
    "Hallucinated",
    "Repetitive",
    "Off-topic",
]
N_RESP = len(RESPONSE_TYPES)

# Reference distribution -- roughly realistic
pi_ref = np.array([0.18, 0.15, 0.12, 0.14, 0.10, 0.12, 0.10, 0.09])
pi_ref = pi_ref / pi_ref.sum()

# Reward values for each response type
rewards = np.array([2.5, 1.2, 2.0, -0.5, 1.0, -2.0, -1.5, -2.5])

beta_bolt = st.slider(
    "Beta (KL penalty weight)",
    min_value=0.01, max_value=5.0, value=1.0, step=0.01,
    key="beta_boltzmann",
)

# Compute optimal tilted distribution
log_pi_star = np.log(pi_ref + 1e-30) + rewards / max(beta_bolt, 1e-10)
log_pi_star -= log_pi_star.max()
pi_star = np.exp(log_pi_star)
pi_star = pi_star / pi_star.sum()

kl_val = kl_divergence(pi_star, pi_ref)

# Regime label
if beta_bolt < 0.1:
    regime_text = "Danger zone -- near-pure reward maximisation. Policy collapses to highest-reward response."
    regime_color = COLORS["red"]
elif beta_bolt > 3.0:
    regime_text = "Very conservative -- policy barely moves from reference. Reward signal is almost ignored."
    regime_color = COLORS["orange"]
else:
    regime_text = "Balanced regime -- policy tilts toward reward while retaining reference diversity."
    regime_color = COLORS["green"]

col_chart, col_info = st.columns([3, 1])

with col_chart:
    fig_bolt = go.Figure()

    fig_bolt.add_trace(go.Bar(
        x=RESPONSE_TYPES,
        y=pi_ref,
        name="pi_ref (reference)",
        marker_color=COLORS["blue"],
        opacity=0.5,
    ))
    fig_bolt.add_trace(go.Bar(
        x=RESPONSE_TYPES,
        y=pi_star,
        name="pi* (optimal RLHF)",
        marker_color=COLORS["green"],
        opacity=0.85,
    ))

    fig_bolt.update_layout(
        barmode="group",
        title=f"Boltzmann Tilt   |   beta = {beta_bolt:.2f}   |   KL(pi* || pi_ref) = {kl_val:.3f} nats",
        xaxis_title="Response Type",
        yaxis_title="Probability",
        height=420,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_bolt, use_container_width=True)

with col_info:
    st.markdown(f"""
<div class="concept-card">
<strong>Rewards</strong><br/>
""" + "<br/>".join(
        f'<span style="color:{COLORS["green"] if r > 0 else COLORS["red"]}">'
        f'{RESPONSE_TYPES[i].replace(chr(10), " ")}: {r:+.1f}</span>'
        for i, r in enumerate(rewards)
    ) + """
</div>
""", unsafe_allow_html=True)

    st.metric("KL divergence", f"{kl_val:.3f} nats")

    st.markdown(f"""
<div class="insight-box" style="border-left-color:{regime_color};">
<strong style="color:{regime_color};">{regime_text}</strong>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PPO CLIPPING DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("3  PPO Clipping Demo")

st.markdown(r"""
The PPO clipped surrogate objective prevents catastrophically large policy
updates:
""")

st.markdown("""
<div class="big-formula">
L<sup>CLIP</sup> = min( r(&theta;) A&#770;, &nbsp;
clip(r(&theta;), 1&minus;&epsilon;, 1+&epsilon;) A&#770; )
</div>
""", unsafe_allow_html=True)

st.markdown(
    "where **r(theta) = pi_theta(a|s) / pi_old(a|s)** is the probability "
    "ratio and **A-hat** is the estimated advantage."
)

epsilon_ppo = st.slider(
    "Epsilon (clip range)",
    min_value=0.05, max_value=0.50, value=0.20, step=0.01,
    key="epsilon_ppo",
)

# Compute the clipped objective over a range of ratios
ratios = np.linspace(0.0, 3.0, 600)

def ppo_clip_objective(r, adv, eps):
    """Compute the PPO clipped objective for a single (r, adv) pair."""
    unclipped = r * adv
    clipped = np.clip(r, 1 - eps, 1 + eps) * adv
    return np.minimum(unclipped, clipped)

loss_pos = np.array([ppo_clip_objective(r, 1.0, epsilon_ppo) for r in ratios])
loss_neg = np.array([ppo_clip_objective(r, -1.0, epsilon_ppo) for r in ratios])

fig_ppo = go.Figure()

# Positive advantage
fig_ppo.add_trace(go.Scatter(
    x=ratios, y=loss_pos,
    name="A-hat > 0 (good action)",
    line=dict(color=COLORS["green"], width=3),
))

# Negative advantage
fig_ppo.add_trace(go.Scatter(
    x=ratios, y=loss_neg,
    name="A-hat < 0 (bad action)",
    line=dict(color=COLORS["red"], width=3),
))

# Clip boundaries
for boundary in [1 - epsilon_ppo, 1 + epsilon_ppo]:
    fig_ppo.add_vline(
        x=boundary, line_dash="dash",
        line_color=COLORS["yellow"], opacity=0.6,
    )

# Shade the clipped region
fig_ppo.add_vrect(
    x0=1 - epsilon_ppo, x1=1 + epsilon_ppo,
    fillcolor=COLORS["yellow"], opacity=0.07,
    annotation_text="trust region",
    annotation_position="top",
    annotation_font_color=COLORS["yellow"],
)

fig_ppo.update_layout(
    title=f"PPO Clipped Objective   |   epsilon = {epsilon_ppo:.2f}",
    xaxis_title="Probability Ratio  r(theta)",
    yaxis_title="Objective L^CLIP",
    height=440,
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
)

st.plotly_chart(fig_ppo, use_container_width=True)

col_pos, col_neg = st.columns(2)
with col_pos:
    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['green']};">Positive Advantage (good action)</strong><br/>
The objective grows with the ratio, but is <em>capped</em> at
r = 1 + &epsilon; = {1 + epsilon_ppo:.2f}.  The policy cannot increase
the probability of a good action by more than &epsilon; in one step.
</div>
""", unsafe_allow_html=True)

with col_neg:
    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['red']};">Negative Advantage (bad action)</strong><br/>
The objective is <em>floored</em> at r = 1 &minus; &epsilon; = {1 - epsilon_ppo:.2f}.
Once the probability ratio drops enough, there is no further incentive
to push it down -- preventing over-correction.
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RLHF TRAINING LOOP SIMULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("4  RLHF Training Loop Simulator")

st.markdown(
    "Step through the six stages of one PPO iteration.  Watch reward climb "
    "and KL divergence grow over ~20 simulated training steps."
)

# Session-state initialisation
if "rlhf_step" not in st.session_state:
    st.session_state.rlhf_step = 0
    st.session_state.rlhf_rewards = []
    st.session_state.rlhf_kls = []
    st.session_state.rlhf_substep = 0
    st.session_state.rlhf_rng = np.random.RandomState(42)

MAX_STEPS = 20

SUBSTEP_LABELS = [
    "Sample prompt from dataset",
    "Generate completion with current policy",
    "Score completion with reward model",
    "Compute advantage estimate (GAE)",
    "Update policy with PPO clipped gradient",
    "Measure KL(pi_theta || pi_ref)",
]

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    run_step = st.button(
        "Run Next Step" if st.session_state.rlhf_step < MAX_STEPS else "Training Complete",
        disabled=st.session_state.rlhf_step >= MAX_STEPS,
        key="rlhf_run_step",
    )
with col_btn2:
    if st.button("Reset Simulation", key="rlhf_reset"):
        st.session_state.rlhf_step = 0
        st.session_state.rlhf_rewards = []
        st.session_state.rlhf_kls = []
        st.session_state.rlhf_substep = 0
        st.session_state.rlhf_rng = np.random.RandomState(42)
        st.rerun()

if run_step and st.session_state.rlhf_step < MAX_STEPS:
    rng = st.session_state.rlhf_rng
    step = st.session_state.rlhf_step

    # Simulate reward: starts ~0.3, rises with diminishing returns + noise
    reward = 0.3 + 1.8 * (1 - np.exp(-0.18 * step)) + rng.normal(0, 0.12)
    # Simulate KL: grows roughly quadratically with some noise
    kl = 0.05 * (step + 1) ** 1.5 + rng.exponential(0.3)

    st.session_state.rlhf_rewards.append(reward)
    st.session_state.rlhf_kls.append(kl)
    st.session_state.rlhf_step += 1
    st.session_state.rlhf_substep = (st.session_state.rlhf_substep + 1) % 6

# Show substep pipeline
if st.session_state.rlhf_step > 0:
    current_sub = st.session_state.rlhf_substep
    pipe_cols = st.columns(6)
    for i, label in enumerate(SUBSTEP_LABELS):
        with pipe_cols[i]:
            if i == current_sub:
                st.markdown(f"""
<div class="concept-card" style="border:2px solid {COLORS['green']};text-align:center;">
<strong style="color:{COLORS['green']};">Step {i+1}</strong><br/>
<span style="font-size:0.82rem;">{label}</span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="concept-card" style="text-align:center;opacity:0.5;">
<strong>Step {i+1}</strong><br/>
<span style="font-size:0.82rem;">{label}</span>
</div>""", unsafe_allow_html=True)

# Charts
if len(st.session_state.rlhf_rewards) > 0:
    steps_arr = np.arange(1, len(st.session_state.rlhf_rewards) + 1)
    rewards_arr = np.array(st.session_state.rlhf_rewards)
    kls_arr = np.array(st.session_state.rlhf_kls)

    col_rew, col_kl = st.columns(2)

    with col_rew:
        fig_rew = go.Figure()
        fig_rew.add_trace(go.Scatter(
            x=steps_arr, y=rewards_arr,
            mode="lines+markers",
            name="Mean Reward",
            line=dict(color=COLORS["green"], width=2),
            marker=dict(size=6),
        ))
        fig_rew.update_layout(
            title="Reward over Training Steps",
            xaxis_title="Step",
            yaxis_title="Mean Reward",
            height=340,
        )
        st.plotly_chart(fig_rew, use_container_width=True)

    with col_kl:
        fig_kl = go.Figure()
        fig_kl.add_trace(go.Scatter(
            x=steps_arr, y=kls_arr,
            mode="lines+markers",
            name="KL(pi_theta || pi_ref)",
            line=dict(color=COLORS["red"], width=2),
            marker=dict(size=6),
        ))
        # Warning threshold band
        fig_kl.add_hrect(
            y0=10, y1=max(20, kls_arr.max() + 2),
            fillcolor=COLORS["red"], opacity=0.08,
            annotation_text="KL danger zone (>10 nats)",
            annotation_position="top left",
            annotation_font_color=COLORS["red"],
        )
        fig_kl.update_layout(
            title="KL Divergence over Training Steps",
            xaxis_title="Step",
            yaxis_title="KL (nats)",
            height=340,
        )
        st.plotly_chart(fig_kl, use_container_width=True)

    # KL alert
    latest_kl = kls_arr[-1]
    if latest_kl > 15:
        st.error(
            f"KL = {latest_kl:.1f} nats -- **training is diverging!**  "
            "The policy is too far from the reference.  In practice you would "
            "increase beta or apply early stopping."
        )
    elif latest_kl > 10:
        st.warning(
            f"KL = {latest_kl:.1f} nats -- entering the danger zone.  "
            "Consider increasing the KL penalty coefficient."
        )

    # Step summary
    st.markdown(f"""
<div class="concept-card">
<strong>Step {st.session_state.rlhf_step} / {MAX_STEPS}</strong> &nbsp;|&nbsp;
Reward = <span style="color:{COLORS['green']};">{rewards_arr[-1]:.3f}</span> &nbsp;|&nbsp;
KL = <span style="color:{COLORS['red']};">{kls_arr[-1]:.3f}</span> nats
</div>
""", unsafe_allow_html=True)
else:
    st.info("Press **Run Next Step** to begin the RLHF training loop simulation.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. WHY PPO IS HARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("5  Why PPO Is Hard")

st.markdown(
    "RLHF with PPO requires **four full models in memory simultaneously**. "
    "For a 70B-parameter base model this is brutal."
)

# Memory breakdown table
models_data = [
    ("Policy (pi_theta)", "70B", "The model being trained", COLORS["blue"]),
    ("Reference (pi_ref)", "70B", "Frozen SFT checkpoint for KL computation", COLORS["cyan"]),
    ("Reward Model", "70B", "Scores completions (can be smaller)", COLORS["green"]),
    ("Value Function", "70B", "Estimates future reward for GAE", COLORS["orange"]),
]

mem_cols = st.columns(4)
for i, (name, params, desc, color) in enumerate(models_data):
    with mem_cols[i]:
        st.markdown(f"""
<div class="concept-card" style="border-left:4px solid {color};text-align:center;">
<strong style="color:{color};font-size:1.1rem;">{name}</strong><br/>
<span style="font-size:1.6rem;font-weight:700;color:{color};">{params}</span><br/>
<span style="font-size:0.82rem;color:{COLORS['gray']};">{desc}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="concept-card" style="border:2px solid {COLORS['red']};text-align:center;">
<strong style="color:{COLORS['red']};font-size:1.2rem;">
Total: 4 x 70B = 280B parameters in memory
</strong><br/>
<span style="color:{COLORS['gray']};">
At fp16 that is ~560 GB -- requiring many A100/H100 GPUs just for inference,
plus optimizer states for the policy and value heads.
</span>
</div>
""", unsafe_allow_html=True)

# Hyperparameter sensitivity panel
st.markdown("#### Hyperparameter Sensitivity")
st.markdown(
    "PPO introduces many sensitive hyperparameters on top of the usual "
    "learning rate and batch size."
)

hyperparam_data = {
    "Parameter": [
        "beta (KL coeff)",
        "epsilon (clip range)",
        "GAE lambda",
        "Learning rate",
        "Num PPO epochs",
        "Batch size",
        "Rollout length",
    ],
    "Typical Range": [
        "0.01 -- 0.5",
        "0.1 -- 0.3",
        "0.9 -- 1.0",
        "1e-6 -- 5e-6",
        "1 -- 4",
        "64 -- 512",
        "128 -- 1024 tokens",
    ],
    "If Too Low": [
        "Reward hacking, mode collapse",
        "Overly aggressive updates",
        "High-bias advantage estimates",
        "No learning",
        "Under-utilised rollouts",
        "High variance gradients",
        "Sparse reward signal",
    ],
    "If Too High": [
        "No learning, stuck at reference",
        "Under-utilised updates",
        "High-variance advantage estimates",
        "Instability, divergence",
        "Overfitting to current batch",
        "Stale rollouts",
        "Expensive generation",
    ],
}

st.dataframe(
    hyperparam_data,
    use_container_width=True,
    hide_index=True,
)

# Visual sensitivity chart -- how reward changes with beta
betas_sweep = np.linspace(0.01, 2.0, 80)
expected_rewards = []
for b in betas_sweep:
    lp = np.log(pi_ref + 1e-30) + rewards / max(b, 1e-10)
    lp -= lp.max()
    p = np.exp(lp)
    p = p / p.sum()
    expected_rewards.append(np.dot(p, rewards))

fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=betas_sweep, y=expected_rewards,
    mode="lines",
    line=dict(color=COLORS["purple"], width=3),
    name="E[r] under pi*",
))
fig_sens.update_layout(
    title="Expected Reward of Optimal Policy vs Beta",
    xaxis_title="Beta (KL penalty weight)",
    yaxis_title="E[r(y)] under pi*",
    height=350,
)
st.plotly_chart(fig_sens, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. KEY INSIGHT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("6  Key Insight")

st.markdown(f"""
<div class="insight-box" style="font-size:1.15rem;padding:24px 28px;">
<strong>"Beta is the exchange rate between reward and KL.
The KL budget is the compression constraint."</strong><br/><br/>
<span style="color:{COLORS['gray']};">
Every nat of KL you spend buys you some expected reward.
Beta sets the price.  A small beta means KL is cheap -- the policy
wanders far from the reference for small reward gains.
A large beta means KL is expensive -- the policy barely moves.
The entire RLHF problem reduces to: <em>given a fixed budget of
distributional change, where should you spend it?</em>
</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="concept-card" style="margin-top:8px;">
<strong>The PPO loop in one sentence:</strong><br/>
Sample prompts &rarr; generate completions &rarr; score with RM &rarr;
compute advantages &rarr; update with clipped gradients &rarr; measure KL
&rarr; repeat.
</div>
""", unsafe_allow_html=True)
