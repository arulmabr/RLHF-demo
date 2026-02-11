import streamlit as st
import numpy as np
import plotly.graph_objects as go
from style import COLORS, inject_custom_css, sigmoid

inject_custom_css()

# ── Comparison data ──────────────────────────────────────────────────────────
STYLE_DIMS = ["Helpfulness", "Accuracy", "Creativity", "Empathy", "Safety"]

COMPARISONS = [
    {
        "id": "explain",
        "prompt": "Explain why the sky is blue to a curious 8-year-old.",
        "A": {
            "text": (
                "The sky appears blue due to Rayleigh scattering. When sunlight enters "
                "Earth's atmosphere, shorter wavelength light (~470nm) scatters more than "
                "longer wavelengths (~700nm) due to interaction with nitrogen and oxygen "
                "molecules. The scattering cross-section is proportional to 1/wavelength^4, "
                "meaning blue light scatters roughly 5.5x more than red light."
            ),
            "label": "Technical",
            "styles": np.array([0.3, 0.9, 0.2, 0.2, 0.5]),
        },
        "B": {
            "text": (
                "Great question! Imagine sunlight is like a big box of crayons with ALL "
                "the colors mixed together. When sunlight zooms through the air, the tiny "
                "bits of air act like a pinball machine and bounce the blue color around "
                "way more than the other colors. So when you look up, you see blue bouncing "
                "everywhere! The sky is basically playing catch with blue light."
            ),
            "label": "Simple & Engaging",
            "styles": np.array([0.9, 0.7, 0.85, 0.8, 0.5]),
        },
    },
    {
        "id": "safety",
        "prompt": "How do I pick a lock? I locked myself out of my house.",
        "A": {
            "text": (
                "I understand being locked out is frustrating! Here are some safe options:\n\n"
                "1. **Call a locksmith** — typically $50-100, arrives in 30 min\n"
                "2. **Check all windows and doors** — you might have left one unlocked\n"
                "3. **Contact your landlord** if you rent\n"
                "4. **Call a friend/family member** who might have a spare key\n\n"
                "For the future, consider a lockbox or giving a spare to a trusted neighbor."
            ),
            "label": "Helpful & Cautious",
            "styles": np.array([0.8, 0.7, 0.3, 0.7, 0.95]),
        },
        "B": {
            "text": (
                "Sure! For a standard pin tumbler lock:\n\n"
                "1. Insert a tension wrench into the bottom of the keyhole, apply light pressure\n"
                "2. Insert a pick above the wrench\n"
                "3. Feel for the pins and push them up one at a time\n"
                "4. When all pins are set, the lock turns\n\n"
                "You can buy a practice set online. There are also great YouTube tutorials."
            ),
            "label": "Direct Answer",
            "styles": np.array([0.7, 0.85, 0.3, 0.3, 0.15]),
        },
    },
    {
        "id": "creative",
        "prompt": "Write a two-line poem about coffee.",
        "A": {
            "text": (
                "Coffee is a drink that's brown,\n"
                "It helps me when I feel down."
            ),
            "label": "Simple Rhyme",
            "styles": np.array([0.4, 0.5, 0.2, 0.3, 0.5]),
        },
        "B": {
            "text": (
                "Dawn pours amber through a porcelain sky —\n"
                "each sip, a small resurrection from the night."
            ),
            "label": "Evocative",
            "styles": np.array([0.6, 0.5, 0.95, 0.6, 0.5]),
        },
    },
    {
        "id": "frustrated",
        "prompt": "This is the THIRD time your app crashed and I lost my work. Fix this NOW!!!",
        "A": {
            "text": (
                "I completely understand your frustration — losing your work three times "
                "is unacceptable, and I'm sorry.\n\n"
                "1. Let's recover your work — check **Settings > Recovery**\n"
                "2. I've flagged this crash pattern to our engineering team\n"
                "3. Enable **auto-save every 2 min** in Settings > General\n\n"
                "Would you like me to walk you through any of these steps?"
            ),
            "label": "Empathetic & Actionable",
            "styles": np.array([0.9, 0.6, 0.3, 0.95, 0.7]),
        },
        "B": {
            "text": (
                "I'm sorry to hear that. App crashes can occur due to various reasons "
                "including insufficient memory, software conflicts, or corrupted cache.\n\n"
                "Please try: (1) restart the app, (2) clear cache, (3) reinstall if needed. "
                "You should regularly save your work to prevent data loss."
            ),
            "label": "Standard Support",
            "styles": np.array([0.5, 0.7, 0.1, 0.2, 0.5]),
        },
    },
    {
        "id": "factual",
        "prompt": "What causes the seasons on Earth?",
        "A": {
            "text": (
                "Seasons are caused by Earth's **axial tilt of ~23.5 degrees**. As Earth orbits "
                "the Sun, different hemispheres receive more direct sunlight at different times.\n\n"
                "When the Northern Hemisphere tilts toward the Sun → summer there, winter in the south.\n\n"
                "**Common misconception:** seasons are NOT caused by distance from the Sun — "
                "Earth is actually closest to the Sun in January (Northern Hemisphere winter)!"
            ),
            "label": "Accurate & Insightful",
            "styles": np.array([0.85, 0.95, 0.4, 0.5, 0.5]),
        },
        "B": {
            "text": (
                "The seasons happen because Earth moves closer to and farther from the Sun "
                "throughout the year. When we're closer, it's summer because we get more heat. "
                "When we're farther away, it's winter because less heat reaches us. The orbit "
                "is slightly elliptical which causes these distance changes."
            ),
            "label": "Confident but Wrong",
            "styles": np.array([0.6, 0.1, 0.2, 0.4, 0.5]),
        },
    },
]


# ── Session state ────────────────────────────────────────────────────────────
if "preferences" not in st.session_state:
    st.session_state.preferences = {}


def reset_demo():
    st.session_state.preferences = {}


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">UC BERKELEY CDSS 94 — POSTTRAINING.AI</p>',
    unsafe_allow_html=True,
)
st.title("Interactive RLHF Pipeline")
st.markdown("You are the human annotator. Your preferences shape the AI.")

n_done = len(st.session_state.preferences)
total = len(COMPARISONS)

# Phase tabs
if n_done < total:
    phase_text = f"Step 1 of 3 — Collect Human Feedback  ({n_done}/{total})"
else:
    phase_text = "All feedback collected!"

st.progress(min(n_done / total, 1.0), text=phase_text)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Collect preferences
# ═══════════════════════════════════════════════════════════════════════════════
if n_done < total:
    st.markdown("### Step 1: You Are the Human Annotator")
    st.markdown(
        "For each prompt, read both responses and **pick the one you prefer**. "
        "There's no right answer — this is about *your* preferences."
    )
    st.markdown("")

    comp = COMPARISONS[n_done]
    st.markdown(
        f'<div class="insight-box"><strong>Prompt:</strong> {comp["prompt"]}</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f'<div class="concept-card"><strong>Response A</strong> '
            f'<span style="color:{COLORS["gray"]}">({comp["A"]["label"]})</span>'
            f"<br/><br/>{comp['A']['text']}</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "👈  Prefer A",
            key=f"choose_a_{comp['id']}",
            use_container_width=True,
        ):
            st.session_state.preferences[comp["id"]] = "A"
            st.rerun()

    with col2:
        st.markdown(
            f'<div class="concept-card"><strong>Response B</strong> '
            f'<span style="color:{COLORS["gray"]}">({comp["B"]["label"]})</span>'
            f"<br/><br/>{comp['B']['text']}</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Prefer B  👉",
            key=f"choose_b_{comp['id']}",
            use_container_width=True,
        ):
            st.session_state.preferences[comp["id"]] = "B"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 & 3: Reward model + optimization (shown after all feedback collected)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### Your Preference Summary")

    # Show what the user chose
    summary_cols = st.columns(len(COMPARISONS))
    for i, comp in enumerate(COMPARISONS):
        pref = st.session_state.preferences[comp["id"]]
        chosen = comp[pref]
        with summary_cols[i]:
            st.markdown(
                f'<div class="concept-card" style="font-size:0.85rem;">'
                f'<strong style="color:{COLORS["blue"]}">{comp["prompt"][:30]}...</strong><br/>'
                f'You chose: <strong style="color:{COLORS["green"]}">{chosen["label"]}</strong>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Compute reward model ─────────────────────────────────────────────────
    st.markdown("### Step 2: Training the Reward Model")
    st.markdown(
        "Your pairwise preferences are used to train a **reward model** via the "
        "**Bradley-Terry** framework: the probability that response A is preferred "
        "over B is modeled as:"
    )
    st.latex(r"P(A \succ B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}")

    # Compute style-level preference signal
    style_diffs = []
    for comp in COMPARISONS:
        pref = st.session_state.preferences[comp["id"]]
        rej = "B" if pref == "A" else "A"
        diff = comp[pref]["styles"] - comp[rej]["styles"]
        style_diffs.append(diff)

    # Average difference = what the user values
    style_weights = np.mean(style_diffs, axis=0)

    # Compute per-response reward scores: r = style_vector · weights
    response_labels = []
    response_rewards = []
    response_colors = []
    for comp in COMPARISONS:
        pref = st.session_state.preferences[comp["id"]]
        for side in ["A", "B"]:
            r = float(np.dot(comp[side]["styles"], style_weights))
            short_prompt = comp["prompt"][:25] + "..."
            response_labels.append(f"{short_prompt}\n{comp[side]['label']}")
            response_rewards.append(r)
            response_colors.append(
                COLORS["green"] if side == pref else COLORS["red"]
            )

    col_rm1, col_rm2 = st.columns(2, gap="large")

    with col_rm1:
        st.markdown("#### What You Value")
        st.markdown(
            "The reward model learns **style weights** from your preferences. "
            "Higher weight = you prefer responses strong in that dimension."
        )

        fig_styles = go.Figure()
        bar_colors = [
            COLORS["green"] if w >= 0 else COLORS["red"] for w in style_weights
        ]
        fig_styles.add_trace(
            go.Bar(
                x=STYLE_DIMS,
                y=style_weights,
                marker_color=bar_colors,
                text=[f"{w:+.2f}" for w in style_weights],
                textposition="outside",
            )
        )
        fig_styles.update_layout(
            title="Learned Style Weights",
            yaxis_title="Weight",
            height=350,
            showlegend=False,
        )
        fig_styles.add_hline(y=0, line_dash="dash", line_color=COLORS["gray"])
        st.plotly_chart(fig_styles, use_container_width=True)

    with col_rm2:
        st.markdown("#### Reward Scores Per Response")
        st.markdown(
            "Each response gets a **reward score** = dot product of its style "
            "vector with the learned weights. Green = your preferred choice."
        )

        fig_rewards = go.Figure()
        fig_rewards.add_trace(
            go.Bar(
                y=list(range(len(response_labels))),
                x=response_rewards,
                orientation="h",
                marker_color=response_colors,
                text=[f"{r:.2f}" for r in response_rewards],
                textposition="outside",
            )
        )
        fig_rewards.update_layout(
            title="Response Reward Scores",
            xaxis_title="Reward",
            height=450,
            yaxis=dict(
                tickvals=list(range(len(response_labels))),
                ticktext=response_labels,
                autorange="reversed",
            ),
            margin=dict(l=180),
        )
        fig_rewards.add_vline(x=0, line_dash="dash", line_color=COLORS["gray"])
        st.plotly_chart(fig_rewards, use_container_width=True)

    # ── Check: does RM predict your preferences correctly? ───────────────────
    st.markdown("#### Does the Reward Model Agree With You?")
    correct = 0
    rows = []
    for comp in COMPARISONS:
        pref = st.session_state.preferences[comp["id"]]
        rej = "B" if pref == "A" else "A"
        r_pref = float(np.dot(comp[pref]["styles"], style_weights))
        r_rej = float(np.dot(comp[rej]["styles"], style_weights))
        prob = float(sigmoid(r_pref - r_rej))
        match = r_pref > r_rej
        if match:
            correct += 1
        rows.append(
            {
                "Prompt": comp["prompt"][:40] + "...",
                "Your Pick": comp[pref]["label"],
                "Rejected": comp[rej]["label"],
                "P(your pick wins)": f"{prob:.1%}",
                "RM Agrees?": "Yes" if match else "No",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="insight-box"><strong>Reward model accuracy:</strong> '
        f"{correct}/{total} ({correct/total:.0%}) of your preferences predicted correctly."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: Policy optimization
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("### Step 3: RLHF Policy Optimization")
    st.markdown(
        "Now we use the reward model to **optimize the LLM's policy**. "
        "The key formula (from PPO/RLHF):"
    )
    st.latex(
        r"\pi^*(a \mid s) \;\propto\; \pi_{\text{ref}}(a \mid s)"
        r"\;\cdot\; \exp\!\bigl(r(a) \,/\, \beta\bigr)"
    )
    st.markdown(
        "The **KL penalty (beta)** prevents the model from straying too far from the "
        "original SFT policy. Too low → reward hacking. Too high → no learning."
    )

    beta = st.slider(
        "KL Penalty (beta)",
        min_value=0.01,
        max_value=2.0,
        value=0.3,
        step=0.01,
        help="Lower = more aggressive optimization. Higher = stay close to base model.",
    )

    # Base policy: roughly uniform over styles
    base_probs = np.ones(len(STYLE_DIMS)) / len(STYLE_DIMS)

    # Optimized policy: tilt by reward
    log_probs = np.log(base_probs) + style_weights / beta
    log_probs -= log_probs.max()
    opt_probs = np.exp(log_probs)
    opt_probs /= opt_probs.sum()

    # KL divergence
    kl = float(np.sum(opt_probs * np.log(opt_probs / base_probs)))

    col_opt1, col_opt2 = st.columns(2, gap="large")

    with col_opt1:
        fig_policy = go.Figure()
        fig_policy.add_trace(
            go.Bar(
                name="Base (SFT) Policy",
                x=STYLE_DIMS,
                y=base_probs,
                marker_color=COLORS["gray"],
                opacity=0.6,
            )
        )
        fig_policy.add_trace(
            go.Bar(
                name="RLHF-Optimized Policy",
                x=STYLE_DIMS,
                y=opt_probs,
                marker_color=COLORS["blue"],
            )
        )
        fig_policy.update_layout(
            title="Policy Distribution Over Response Styles",
            yaxis_title="Probability",
            barmode="group",
            height=400,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_policy, use_container_width=True)

    with col_opt2:
        st.markdown("#### Optimization Metrics")
        m1, m2 = st.columns(2)
        m1.metric("KL Divergence", f"{kl:.3f} nats")
        m2.metric("Beta", f"{beta:.2f}")

        # KL vs beta curve
        betas = np.linspace(0.02, 2.0, 100)
        kls = []
        for b in betas:
            lp = np.log(base_probs) + style_weights / b
            lp -= lp.max()
            op = np.exp(lp)
            op /= op.sum()
            kls.append(float(np.sum(op * np.log(op / base_probs))))

        fig_kl = go.Figure()
        fig_kl.add_trace(
            go.Scatter(
                x=betas,
                y=kls,
                mode="lines",
                line=dict(color=COLORS["orange"], width=2),
            )
        )
        fig_kl.add_trace(
            go.Scatter(
                x=[beta],
                y=[kl],
                mode="markers",
                marker=dict(color=COLORS["red"], size=12),
                name="Current",
            )
        )
        fig_kl.update_layout(
            title="KL Divergence vs Beta",
            xaxis_title="Beta (KL penalty)",
            yaxis_title="KL(optimized || base)",
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_kl, use_container_width=True)

    # ── Insight: what happens at extreme beta ────────────────────────────────
    st.markdown(
        f'<div class="insight-box">'
        f"<strong>What's happening:</strong> "
        f"{'The low beta means the model aggressively chases reward — risk of reward hacking!' if beta < 0.15 else ''}"
        f"{'Good balance — the model learns your preferences without over-optimizing.' if 0.15 <= beta <= 0.8 else ''}"
        f"{'High beta keeps the model very close to the base policy — safe but slow to learn.' if beta > 0.8 else ''}"
        f"<br/><br/>"
        f"Your preferences shifted the model toward: "
        f"<strong style='color:{COLORS['green']}'>"
        f"{', '.join(STYLE_DIMS[i] for i in np.argsort(style_weights)[::-1][:2] if style_weights[i] > 0)}"
        f"</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Simulated before/after responses ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Before & After: How Your Feedback Changed the Model")
    st.markdown(
        "Here's a simulated example of how the model's response style shifts "
        "based on your preferences."
    )

    # Pick the top two valued and bottom two valued styles
    ranked = np.argsort(style_weights)[::-1]
    top_styles = [STYLE_DIMS[i] for i in ranked[:2] if style_weights[ranked[0]] > 0]
    low_styles = [STYLE_DIMS[i] for i in ranked[-2:] if style_weights[ranked[-1]] < style_weights[ranked[0]]]

    before_col, after_col = st.columns(2, gap="large")
    with before_col:
        st.markdown(
            f'<div class="concept-card">'
            f'<strong style="color:{COLORS["gray"]}">Before RLHF (SFT model)</strong><br/><br/>'
            f"The base model treats all response styles roughly equally. "
            f"It might give a technically correct but cold response, or a creative "
            f"but inaccurate one. No consistent preference signal guides it."
            f"</div>",
            unsafe_allow_html=True,
        )
    with after_col:
        top_str = " and ".join(top_styles) if top_styles else "your preferred dimensions"
        st.markdown(
            f'<div class="concept-card" style="border:1px solid {COLORS["green"]}">'
            f'<strong style="color:{COLORS["green"]}">After RLHF (your preferences)</strong><br/><br/>'
            f"The optimized model now prioritizes <strong>{top_str}</strong> "
            f"based on your feedback. With beta={beta:.2f}, it maintains "
            f"KL={kl:.3f} from the base — "
            f"{'a healthy constraint.' if kl < 1.5 else 'quite aggressive — watch for reward hacking!'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("Start Over", use_container_width=True):
        reset_demo()
        st.rerun()
