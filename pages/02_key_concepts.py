import streamlit as st

st.set_page_config(page_title="Key Concepts", page_icon="📐", layout="wide")

import numpy as np
import plotly.graph_objects as go
from style import COLORS, inject_custom_css, softmax, kl_divergence, sigmoid

inject_custom_css()

st.markdown(
    '<p class="section-header">CORE THEORY</p>',
    unsafe_allow_html=True,
)
st.title("Key Concepts")
st.markdown("The ideas that make RLHF work — and make it hard.")

tab1, tab2, tab3 = st.tabs(["KL Divergence", "Reward Hacking", "DPO vs RLHF"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: KL Divergence
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### KL Divergence: The Leash on RLHF")
    st.markdown(
        "KL divergence measures how far one distribution has drifted from another. "
        "In RLHF, it's the **penalty** that keeps the optimized policy close to the "
        "base SFT model."
    )
    st.latex(
        r"D_{\text{KL}}(P \| Q) = \sum_x P(x) \ln \frac{P(x)}{Q(x)}"
    )

    st.markdown("#### Interactive KL Calculator")
    st.markdown("Adjust the optimized distribution and see KL change in real time.")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Simple 4-token distribution
        tokens = ["good", "great", "bad", "terrible"]
        st.markdown("**Base (SFT) distribution:**")
        base = np.array([0.4, 0.3, 0.2, 0.1])
        for tok, p in zip(tokens, base):
            st.markdown(f"- `{tok}`: {p:.1%}")

        st.markdown("**Optimized distribution:**")
        p_good = st.slider("P(good)", 0.01, 0.97, 0.5, 0.01)
        max_great = max(0.97 - p_good, 0.02)
        p_great = st.slider("P(great)", 0.01, max_great, min(0.3, max_great), 0.01)
        max_bad = max(0.97 - p_good - p_great, 0.02)
        p_bad = st.slider("P(bad)", 0.01, max_bad, min(0.15, max_bad), 0.01)
        p_terrible = max(1.0 - p_good - p_great - p_bad, 0.01)
        opt = np.array([p_good, p_great, p_bad, p_terrible])
        opt = opt / opt.sum()  # normalize

    with col2:
        kl_val = kl_divergence(opt, base)
        reverse_kl = kl_divergence(base, opt)

        st.metric("Forward KL: D(optimized || base)", f"{kl_val:.4f} nats")
        st.metric("Reverse KL: D(base || optimized)", f"{reverse_kl:.4f} nats")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Base (SFT)",
                x=tokens,
                y=base,
                marker_color=COLORS["gray"],
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Optimized",
                x=tokens,
                y=opt,
                marker_color=COLORS["blue"],
            )
        )
        fig.update_layout(
            title=f"KL = {kl_val:.4f} nats",
            yaxis_title="Probability",
            barmode="group",
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        f'<div class="insight-box">'
        f"<strong>Why it matters:</strong> "
        f"Without KL penalty, RLHF would collapse all probability onto the single "
        f"highest-reward token. The KL term says: 'optimize for reward, but don't "
        f"forget what you learned in SFT.' A typical KL budget in practice is 5-15 nats."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Forward vs Reverse KL
    st.markdown("---")
    st.markdown("#### Forward vs Reverse KL")

    fwd_col, rev_col = st.columns(2)
    with fwd_col:
        st.markdown(
            f'<div class="concept-card">'
            f'<strong style="color:{COLORS["blue"]}">Forward KL: D(P || Q)</strong><br/><br/>'
            f"<strong>Mean-seeking.</strong> P spreads to cover everywhere Q has mass. "
            f"If Q is multimodal, P tries to cover all modes (even if it means "
            f"putting probability in low-density regions between modes).<br/><br/>"
            f"Used in: variational inference, supervised learning."
            f"</div>",
            unsafe_allow_html=True,
        )
    with rev_col:
        st.markdown(
            f'<div class="concept-card">'
            f'<strong style="color:{COLORS["orange"]}">Reverse KL: D(Q || P)</strong><br/><br/>'
            f"<strong>Mode-seeking.</strong> Q concentrates on one mode of P and ignores "
            f"the rest. Produces sharper, more confident outputs but may miss diversity.<br/><br/>"
            f"Used in: RLHF, policy optimization, GANs."
            f"</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Reward Hacking
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Goodhart's Law & Reward Hacking")
    st.markdown(
        '*"When a measure becomes a target, it ceases to be a good measure."*\n\n'
        "The reward model is a **proxy** for what humans actually want. "
        "Over-optimize against it and the model finds exploits."
    )

    st.markdown("#### Simulation: Optimizing Against a Proxy Reward")
    st.markdown(
        "Imagine the *true* quality of a response peaks at some optimum, but the "
        "reward model (proxy) keeps going up because it can't capture every nuance."
    )

    sl1, sl2 = st.columns(2)
    with sl1:
        noise = st.slider("Proxy noise (misalignment)", 0.0, 1.0, 0.3, 0.05)
    with sl2:
        kl_budget = st.slider("KL budget", 0.5, 20.0, 5.0, 0.5)

    x = np.linspace(0, 20, 200)

    # True reward: peaks then drops
    true_reward = 3 * np.exp(-((x - 5) ** 2) / 8) - 0.02 * (x - 5) ** 2

    # Proxy reward: keeps climbing (misaligned)
    proxy_reward = true_reward + noise * 0.4 * x

    # Policy optimizes proxy up to KL budget
    opt_point = min(kl_budget, 20)
    policy_x = opt_point

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=true_reward,
            mode="lines",
            name="True Quality",
            line=dict(color=COLORS["green"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=proxy_reward,
            mode="lines",
            name="Proxy Reward (RM)",
            line=dict(color=COLORS["red"], width=3, dash="dash"),
        )
    )
    # Mark the optimal point and policy point with offset annotations
    fig.add_vline(x=5, line_dash="dot", line_color=COLORS["gray"],
                   annotation_text="True optimum",
                   annotation_position="top left")
    fig.add_vline(x=policy_x, line_dash="dot", line_color=COLORS["orange"],
                   annotation_text=f"Policy (KL={kl_budget:.1f})",
                   annotation_position="top right")

    fig.update_layout(
        title="Reward Hacking: Proxy vs True Quality",
        xaxis_title="Optimization pressure (KL from base)",
        yaxis_title="Reward",
        height=420,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(b=80),
    )
    st.plotly_chart(fig, width="stretch")

    # Evaluation
    true_at_policy = float(np.interp(policy_x, x, true_reward))
    proxy_at_policy = float(np.interp(policy_x, x, proxy_reward))
    true_optimum = float(np.max(true_reward))

    c1, c2, c3 = st.columns(3)
    c1.metric("True quality at policy", f"{true_at_policy:.2f}")
    c2.metric("Proxy reward at policy", f"{proxy_at_policy:.2f}")
    c3.metric("True optimum", f"{true_optimum:.2f}")

    gap = true_optimum - true_at_policy
    st.markdown(
        f'<div class="insight-box">'
        f"<strong>{'Reward hacking detected!' if gap > 0.5 else 'Looking good.'}</strong> "
        f"{'The model over-optimized the proxy and lost true quality. This is why KL budgets matter.' if gap > 0.5 else 'The KL budget is keeping the model near the true optimum.'}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DPO vs RLHF
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### DPO: Skipping the Reward Model")
    st.markdown(
        "**Direct Preference Optimization (DPO)** reformulates RLHF so you don't "
        "need a separate reward model. Instead, preferences directly become a "
        "supervised loss on the policy."
    )

    st.markdown("#### The DPO Loss")
    st.latex(
        r"\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left("
        r"\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} "
        r"- \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}"
        r"\right)\right]"
    )
    st.markdown(
        "Where y_w = preferred response, y_l = rejected response, "
        "and beta controls regularization strength."
    )

    # ── DPO gradient visualization ──
    st.markdown("#### DPO Gradient: When Does Learning Happen?")
    st.markdown(
        "The DPO gradient is weighted by how **wrong** the implicit reward model is. "
        "Adjust the log-ratio differences to see when the gradient is strongest."
    )

    dpo_beta = st.slider("Beta (regularization)", 0.05, 2.0, 0.1, 0.05, key="dpo_beta")

    # Simulate: range of log-ratio differences
    margin = np.linspace(-4, 4, 200)
    # DPO gradient weight = sigma(-beta * margin) where margin = log_ratio_w - log_ratio_l
    grad_weight = sigmoid(-dpo_beta * margin)

    fig_dpo = go.Figure()
    fig_dpo.add_trace(
        go.Scatter(
            x=margin,
            y=grad_weight,
            mode="lines",
            line=dict(color=COLORS["purple"], width=3),
            fill="tozeroy",
            fillcolor=f"rgba(155, 89, 182, 0.2)",
        )
    )
    fig_dpo.add_vline(x=0, line_dash="dash", line_color=COLORS["gray"])
    fig_dpo.update_layout(
        title=f"DPO Gradient Weight (beta={dpo_beta})",
        xaxis_title="Implicit reward margin (preferred - rejected)",
        yaxis_title="Gradient weight",
        height=350,
    )
    fig_dpo.add_annotation(
        x=-2, y=0.85,
        text="Strong gradient<br>(model is wrong)",
        showarrow=False,
        font=dict(color=COLORS["red"]),
    )
    fig_dpo.add_annotation(
        x=2, y=0.15,
        text="Weak gradient<br>(model already right)",
        showarrow=False,
        font=dict(color=COLORS["green"]),
    )
    st.plotly_chart(fig_dpo, width="stretch")

    # ── Comparison table ──
    st.markdown("---")
    st.markdown("#### RLHF vs DPO: Tradeoffs")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown(
            f'<div class="concept-card" style="border-left: 4px solid {COLORS["blue"]}">'
            f'<strong style="color:{COLORS["blue"]}">RLHF (PPO)</strong><br/><br/>'
            f"<strong>Pros:</strong><br/>"
            f"- Reward model generalizes to new prompts<br/>"
            f"- Can iterate and improve reward model<br/>"
            f"- More flexible optimization<br/><br/>"
            f"<strong>Cons:</strong><br/>"
            f"- Complex: 4 models in memory (policy, ref, RM, critic)<br/>"
            f"- Unstable training (PPO hyperparams)<br/>"
            f"- Reward hacking risk<br/>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with comp_col2:
        st.markdown(
            f'<div class="concept-card" style="border-left: 4px solid {COLORS["purple"]}">'
            f'<strong style="color:{COLORS["purple"]}">DPO</strong><br/><br/>'
            f"<strong>Pros:</strong><br/>"
            f"- Simple: supervised loss, 2 models (policy, ref)<br/>"
            f"- Stable training<br/>"
            f"- No reward model to hack<br/><br/>"
            f"<strong>Cons:</strong><br/>"
            f"- Doesn't generalize beyond training pairs<br/>"
            f"- Implicit reward may not capture complexity<br/>"
            f"- Sensitive to beta choice<br/>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="insight-box">'
        f"<strong>In practice:</strong> DPO is increasingly popular for its simplicity. "
        f"Many production systems use DPO for initial alignment, then RLHF/PPO for "
        f"fine-grained optimization. The field is still evolving — newer methods like "
        f"GRPO and iterative DPO continue to push the frontier."
        f"</div>",
        unsafe_allow_html=True,
    )
