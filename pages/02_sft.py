"""
Page 02 -- Supervised Fine-Tuning (SFT)
The simplest post-training technique: train on (prompt, ideal_response) pairs.
"""

from style import inject_custom_css, COLORS, softmax, kl_divergence, entropy
import streamlit as st
import numpy as np
import plotly.graph_objects as go

inject_custom_css()

# ── Page header ──────────────────────────────────────────────────────────────

st.markdown(
    '<p class="section-header">SECTION II &mdash; SUPERVISED FINE-TUNING</p>',
    unsafe_allow_html=True,
)
st.title("SFT -- The Simplest Post-Training")

st.markdown(
    """
SFT is **next-token prediction on curated data**.  You collect
*(prompt, ideal_response)* pairs, then maximize the log-likelihood of the
ideal tokens -- but **only on the assistant tokens** (the prompt is masked).

This is equivalent to **forward-KL minimization**: you are fitting
$\\pi_\\theta$ to cover the modes of the target distribution defined by
your curated dataset.
"""
)

st.markdown(
    """<div class="big-formula">
L<sub>SFT</sub> &nbsp;=&nbsp; &minus; E<sub>(x, y) ~ D</sub>
&bigl[ &sum;<sub>t</sub> log &pi;<sub>&theta;</sub>(y<sub>t</sub> | x, y<sub>&lt;t</sub>) &bigr;]
</div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DISTRIBUTION SHIFT VISUALISER
# ═══════════════════════════════════════════════════════════════════════════

st.header("1. Distribution Shift Visualizer")

st.markdown(
    """
The base (pretrained) model assigns probability mass broadly across many
response *types*.  SFT concentrates that mass onto the desired behaviors
(helpful, factual) and away from undesirable ones (harmful, sycophantic).
"""
)

CATEGORIES = [
    "Helpful",
    "Harmful",
    "Creative",
    "Factual",
    "Sycophantic",
    "Terse",
    "Verbose",
    "Refusal",
]

# Raw logits for the base model -- fairly uniform with slight variance
BASE_LOGITS = np.array([1.0, 0.8, 1.1, 0.9, 0.7, 0.6, 0.9, 0.5])

# Target SFT direction: boost helpful/factual/creative, suppress harmful/sycophantic
SFT_DIRECTION = np.array([3.0, -3.0, 1.5, 2.5, -2.0, -0.5, -0.5, 0.8])

col_a, col_b = st.columns(2)
with col_a:
    n_examples = st.slider(
        "Number of SFT examples",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=100,
        key="sft_n_examples",
    )
with col_b:
    data_quality = st.slider(
        "Data quality",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        key="sft_data_quality",
    )

# Blend strength grows with log(n) and linearly with quality
blend = data_quality * np.clip(np.log10(n_examples) / np.log10(100_000), 0, 1)
sft_logits = BASE_LOGITS + blend * SFT_DIRECTION

base_probs = softmax(BASE_LOGITS, temperature=1.0)
sft_probs = softmax(sft_logits, temperature=1.0)

fig_dist = go.Figure()
fig_dist.add_trace(
    go.Bar(
        x=CATEGORIES,
        y=base_probs,
        name="Base Model",
        marker_color=COLORS["gray"],
        opacity=0.55,
    )
)
fig_dist.add_trace(
    go.Bar(
        x=CATEGORIES,
        y=sft_probs,
        name="After SFT",
        marker_color=COLORS["blue"],
    )
)
fig_dist.update_layout(
    barmode="group",
    title="Response-Type Distribution: Base vs SFT",
    yaxis_title="Probability",
    xaxis_title="Response Category",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig_dist, use_container_width=True)

# Show KL and entropy as live metrics
kl_val = kl_divergence(sft_probs, base_probs)
ent_base = entropy(base_probs)
ent_sft = entropy(sft_probs)

m1, m2, m3 = st.columns(3)
m1.metric("KL(SFT || Base)", f"{kl_val:.3f} nats")
m2.metric("Entropy (Base)", f"{ent_base:.2f} bits")
m3.metric("Entropy (SFT)", f"{ent_sft:.2f} bits")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  QUALITY vs QUANTITY  (The LIMA Insight)
# ═══════════════════════════════════════════════════════════════════════════

st.header("2. Quality vs Quantity -- The LIMA Insight")

st.markdown(
    """
The **LIMA** paper showed that just 1,000 carefully curated examples can
rival models trained on orders of magnitude more noisy data.
The takeaway: **quality scales linearly; quantity saturates fast.**
"""
)

col_q1, col_q2 = st.columns(2)
with col_q1:
    ds_size = st.slider(
        "Dataset size",
        min_value=1_000,
        max_value=1_000_000,
        value=50_000,
        step=1_000,
        key="lima_ds",
    )
with col_q2:
    quality_val = st.slider(
        "Data quality (0 = noisy, 1 = perfect curation)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="lima_quality",
    )

# Build the full surface for the contour, and mark the user point
sizes = np.logspace(3, 6, 80)  # 1K  -> 1M
qualities = np.linspace(0, 1, 80)
S, Q = np.meshgrid(sizes, qualities)

# Model quality: fast log-saturation with size, linear with quality
model_quality = Q * 0.7 + 0.3 * (np.log10(S) - 3) / 3
model_quality = np.clip(model_quality, 0, 1)

fig_lima = go.Figure()
fig_lima.add_trace(
    go.Contour(
        x=np.log10(sizes),
        y=qualities,
        z=model_quality,
        colorscale=[
            [0.0, COLORS["bg"]],
            [0.35, COLORS["purple"]],
            [0.65, COLORS["blue"]],
            [1.0, COLORS["green"]],
        ],
        contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
        colorbar=dict(title="Quality"),
        hovertemplate="log10(size)=%{x:.1f}<br>quality=%{y:.2f}<br>score=%{z:.2f}<extra></extra>",
    )
)

# Mark the user's chosen point
user_score = quality_val * 0.7 + 0.3 * (np.log10(ds_size) - 3) / 3
user_score = float(np.clip(user_score, 0, 1))
fig_lima.add_trace(
    go.Scatter(
        x=[np.log10(ds_size)],
        y=[quality_val],
        mode="markers+text",
        marker=dict(size=14, color=COLORS["orange"], symbol="x", line=dict(width=2, color="white")),
        text=[f"score={user_score:.2f}"],
        textposition="top center",
        textfont=dict(color=COLORS["orange"], size=13),
        showlegend=False,
    )
)

fig_lima.update_layout(
    title="Model Quality = f(dataset size, data quality)",
    xaxis_title="log10( Dataset Size )",
    yaxis_title="Data Quality",
    height=460,
)
st.plotly_chart(fig_lima, use_container_width=True)

st.markdown(
    f"""<div class="insight-box">
<strong>Your setting:</strong> {ds_size:,} examples at quality {quality_val:.2f}
&rarr; model-quality score = <strong>{user_score:.2f}</strong><br/>
Notice: moving the <em>quality</em> slider has a much bigger effect than
moving the <em>size</em> slider once you pass ~10K examples.
</div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  FAILURE MODES PANEL
# ═══════════════════════════════════════════════════════════════════════════

st.header("3. Failure Modes of SFT")

st.markdown(
    """
SFT is simple, but it is easy to get wrong.  The four most common failure
modes each have a characteristic signature you can see in training curves.
"""
)

tab_forget, tab_overfit, tab_syco, tab_epoch = st.tabs(
    [
        "Catastrophic Forgetting",
        "Format Overfitting",
        "Sycophancy",
        "1-Epoch Phenomenon",
    ]
)

# ── 3a. Catastrophic Forgetting ──────────────────────────────────────────

with tab_forget:
    st.subheader("Catastrophic Forgetting")
    st.markdown(
        """
    A learning rate that is too high overwrites the pretrained weights.
    The model may learn the SFT format quickly but **lose general knowledge**.
    """
    )

    lr_exp = st.slider(
        "Learning rate (exponent: 10^x)",
        min_value=-6.0,
        max_value=-3.0,
        value=-5.0,
        step=0.1,
        key="lr_forget",
    )
    lr_val = 10 ** lr_exp

    steps = np.arange(0, 5001, 50)
    # SFT capability rises; pretrained capability decays with high LR
    sft_capability = 1.0 - np.exp(-steps / (800 / (lr_val * 1e5)))
    pretrained_cap = np.exp(-steps * lr_val * 0.4)
    # Combined score
    combined = 0.5 * sft_capability + 0.5 * pretrained_cap

    fig_forget = go.Figure()
    fig_forget.add_trace(
        go.Scatter(x=steps, y=sft_capability, name="SFT Task Quality", line=dict(color=COLORS["green"], width=2))
    )
    fig_forget.add_trace(
        go.Scatter(x=steps, y=pretrained_cap, name="Pretrained Knowledge", line=dict(color=COLORS["red"], width=2))
    )
    fig_forget.add_trace(
        go.Scatter(
            x=steps,
            y=combined,
            name="Overall Quality",
            line=dict(color=COLORS["orange"], width=2, dash="dash"),
        )
    )
    fig_forget.update_layout(
        title=f"Catastrophic Forgetting  (LR = {lr_val:.1e})",
        xaxis_title="Training Steps",
        yaxis_title="Capability (0-1)",
        yaxis_range=[-0.05, 1.1],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_forget, use_container_width=True)

    st.markdown(
        f"""<div class="insight-box">
    <strong>LR = {lr_val:.1e}:</strong> {"Safe range -- pretrained knowledge preserved." if lr_val <= 3e-5 else "Danger zone -- pretrained capabilities are being destroyed."}
    </div>""",
        unsafe_allow_html=True,
    )

# ── 3b. Format Overfitting ──────────────────────────────────────────────

with tab_overfit:
    st.subheader("Format Overfitting / Memorization")
    st.markdown(
        """
    Training for too many epochs causes the model to **memorize** the SFT
    dataset instead of generalizing.  After ~1-2 epochs the validation
    quality plateaus and then degrades.
    """
    )

    n_epochs = st.slider(
        "Number of epochs",
        min_value=1,
        max_value=10,
        value=3,
        key="overfit_epochs",
    )

    epoch_arr = np.linspace(0, 10, 200)
    train_loss = 2.5 * np.exp(-1.2 * epoch_arr) + 0.3 * np.exp(-0.15 * epoch_arr)
    val_quality = 1.0 - 0.6 * np.exp(-1.5 * epoch_arr) - 0.04 * np.maximum(epoch_arr - 2, 0) ** 1.3
    val_quality = np.clip(val_quality, 0, 1)

    fig_overfit = go.Figure()
    fig_overfit.add_trace(
        go.Scatter(x=epoch_arr, y=train_loss, name="Train Loss", line=dict(color=COLORS["blue"], width=2))
    )
    fig_overfit.add_trace(
        go.Scatter(
            x=epoch_arr,
            y=val_quality,
            name="Validation Quality",
            line=dict(color=COLORS["green"], width=2),
            yaxis="y2",
        )
    )
    # Vertical line at chosen epoch
    fig_overfit.add_vline(
        x=n_epochs,
        line_dash="dash",
        line_color=COLORS["orange"],
        annotation_text=f"Stop @ epoch {n_epochs}",
        annotation_position="top right",
        annotation_font_color=COLORS["orange"],
    )
    # Sweet-spot zone
    fig_overfit.add_vrect(x0=1, x1=2, fillcolor=COLORS["green"], opacity=0.08, line_width=0)

    fig_overfit.update_layout(
        title="Train Loss & Validation Quality vs Epoch",
        xaxis_title="Epoch",
        yaxis=dict(title="Train Loss", side="left"),
        yaxis2=dict(title="Val Quality", overlaying="y", side="right", range=[0, 1.05]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_overfit, use_container_width=True)

    # Report val quality at the chosen epoch
    idx = np.argmin(np.abs(epoch_arr - n_epochs))
    vq = val_quality[idx]
    st.markdown(
        f"""<div class="insight-box">
    At <strong>epoch {n_epochs}</strong> the validation quality is <strong>{vq:.2f}</strong>.
    {"The green band (epochs 1-2) is the sweet spot." if n_epochs <= 2 else "You are past the sweet spot -- the model is likely memorising formatting patterns."}
    </div>""",
        unsafe_allow_html=True,
    )

# ── 3c. Sycophancy ──────────────────────────────────────────────────────

with tab_syco:
    st.subheader("Sycophancy from Training Data Bias")
    st.markdown(
        """
    If the SFT dataset contains a high *agreement rate* -- i.e., the
    assistant almost always agrees with the user -- the model learns to
    be **sycophantic**.  It will tell users what they want to hear, even
    when they are wrong.
    """
    )

    agree_rate = st.slider(
        "Agreement rate in training data",
        min_value=0.5,
        max_value=1.0,
        value=0.75,
        step=0.01,
        key="syco_rate",
    )

    # Simulate model behavior distribution
    user_correct_pct = np.linspace(0, 1, 50)  # fraction of time user is correct
    # A perfectly calibrated model agrees when user is right, disagrees when wrong
    ideal_agreement = user_correct_pct
    # A sycophantic model agrees more than it should
    syco_agreement = agree_rate * np.ones_like(user_correct_pct)
    # Blend: model learns something in between but biased toward sycophancy
    model_agreement = 0.3 * ideal_agreement + 0.7 * syco_agreement

    fig_syco = go.Figure()
    fig_syco.add_trace(
        go.Scatter(
            x=user_correct_pct,
            y=ideal_agreement,
            name="Ideal (calibrated)",
            line=dict(color=COLORS["green"], width=2, dash="dash"),
        )
    )
    fig_syco.add_trace(
        go.Scatter(
            x=user_correct_pct,
            y=model_agreement,
            name="SFT Model",
            line=dict(color=COLORS["red"], width=2),
            fill="tonexty",
            fillcolor="rgba(231,76,60,0.12)",
        )
    )
    fig_syco.update_layout(
        title=f"Agreement Rate: Ideal vs SFT Model  (data agree rate = {agree_rate:.0%})",
        xaxis_title="Fraction of time user is actually correct",
        yaxis_title="Model agreement rate",
        yaxis_range=[0, 1.05],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_syco, use_container_width=True)

    syco_gap = float(np.mean(model_agreement - ideal_agreement))
    st.markdown(
        f"""<div class="insight-box">
    The <span style="color:{COLORS['red']};">red shaded area</span> is the
    <strong>sycophancy gap</strong> (mean = {syco_gap:.2f}).
    {"Minimal sycophancy -- the dataset is well-balanced." if agree_rate < 0.65 else "Significant sycophancy risk -- consider adding adversarial / disagreement examples."}
    </div>""",
        unsafe_allow_html=True,
    )

# ── 3d. 1-Epoch Phenomenon ──────────────────────────────────────────────

with tab_epoch:
    st.subheader("The 1-Epoch Phenomenon")
    st.markdown(
        """
    Empirically, the best checkpoint is almost always between **0.5 and 1.5
    epochs**.  Training loss keeps falling beyond that, but *held-out
    evaluation quality* does not improve -- and often degrades.

    This is one of the most robust findings in SFT practice.
    """
    )

    epoch_range = np.linspace(0, 6, 300)
    train_l = 3.0 * np.exp(-1.8 * epoch_range) + 0.2 * np.exp(-0.3 * epoch_range) + 0.15
    # Validation quality: sharp rise, peak near 1, slow decay
    val_q = 1.0 - 0.75 * np.exp(-2.5 * epoch_range) - 0.035 * np.maximum(epoch_range - 1.0, 0) ** 1.4
    val_q = np.clip(val_q, 0, 1)

    fig_epoch = go.Figure()
    fig_epoch.add_trace(
        go.Scatter(
            x=epoch_range,
            y=train_l,
            name="Train Loss",
            line=dict(color=COLORS["blue"], width=2),
        )
    )
    fig_epoch.add_trace(
        go.Scatter(
            x=epoch_range,
            y=val_q,
            name="Validation Quality",
            line=dict(color=COLORS["green"], width=2),
            yaxis="y2",
        )
    )

    # Mark the peak
    peak_idx = int(np.argmax(val_q))
    peak_ep = epoch_range[peak_idx]
    peak_vq = val_q[peak_idx]
    fig_epoch.add_trace(
        go.Scatter(
            x=[peak_ep],
            y=[peak_vq],
            mode="markers+text",
            marker=dict(size=12, color=COLORS["orange"], symbol="star"),
            text=[f"Best @ {peak_ep:.1f} ep"],
            textposition="top center",
            textfont=dict(color=COLORS["orange"], size=12),
            showlegend=False,
            yaxis="y2",
        )
    )

    fig_epoch.add_vrect(x0=0.5, x1=1.5, fillcolor=COLORS["green"], opacity=0.07, line_width=0)

    fig_epoch.update_layout(
        title="The 1-Epoch Phenomenon",
        xaxis_title="Epoch",
        yaxis=dict(title="Train Loss", side="left"),
        yaxis2=dict(title="Validation Quality", overlaying="y", side="right", range=[0, 1.05]),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_epoch, use_container_width=True)

    st.markdown(
        f"""<div class="insight-box">
    <strong>Best checkpoint:</strong> epoch {peak_ep:.1f} &nbsp;(val quality = {peak_vq:.2f}).<br/>
    The green band marks the empirical sweet spot (0.5 -- 1.5 epochs).
    Training loss continues to drop beyond this, but validation quality does not follow.
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# KEY INSIGHT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """<div class="insight-box" style="font-size:1.15rem; text-align:center; padding:24px;">
<strong>Key Insight:</strong>&ensp;Quality &gt;&gt; Quantity.&ensp;
The craft of SFT is the craft of <em>dataset curation</em>.
</div>""",
    unsafe_allow_html=True,
)
