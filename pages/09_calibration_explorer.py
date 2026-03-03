import streamlit as st

st.set_page_config(page_title="How Confident?", page_icon="🎯", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">EVALS &amp; MODEL BEHAVIOR</p>', unsafe_allow_html=True)
st.title("How Confident?")
st.markdown("Can you trust a model that says *'I'm 95% sure'*? Explore calibration — the gap between confidence and accuracy.")
st.markdown("---")

# ── Question bank ───────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "question": "Who received the IEEE Frank Rosenblatt Award in 2010?",
        "model_answer": "Geoffrey Hinton",
        "model_confidence": 90,
        "actual_answer": "Michio Sugeno",
        "is_correct": False,
    },
    {
        "question": "What is the capital of Australia?",
        "model_answer": "Canberra",
        "model_confidence": 85,
        "actual_answer": "Canberra",
        "is_correct": True,
    },
    {
        "question": "In what year was the Eiffel Tower completed?",
        "model_answer": "1889",
        "model_confidence": 95,
        "actual_answer": "1889",
        "is_correct": True,
    },
    {
        "question": "Who painted the Mona Lisa?",
        "model_answer": "Leonardo da Vinci",
        "model_confidence": 98,
        "actual_answer": "Leonardo da Vinci",
        "is_correct": True,
    },
    {
        "question": "What is the atomic number of gold?",
        "model_answer": "79",
        "model_confidence": 92,
        "actual_answer": "79",
        "is_correct": True,
    },
    {
        "question": "Who wrote '1984'?",
        "model_answer": "Aldous Huxley",
        "model_confidence": 78,
        "actual_answer": "George Orwell",
        "is_correct": False,
    },
    {
        "question": "What is the speed of light in km/s?",
        "model_answer": "Approximately 300,000 km/s",
        "model_confidence": 88,
        "actual_answer": "Approximately 299,792 km/s",
        "is_correct": True,
    },
    {
        "question": "Who was the first woman to win a Nobel Prize?",
        "model_answer": "Marie Curie",
        "model_confidence": 94,
        "actual_answer": "Marie Curie",
        "is_correct": True,
    },
    {
        "question": "What year did the Berlin Wall fall?",
        "model_answer": "1991",
        "model_confidence": 72,
        "actual_answer": "1989",
        "is_correct": False,
    },
    {
        "question": "What is the largest bone in the human body?",
        "model_answer": "The tibia",
        "model_confidence": 81,
        "actual_answer": "The femur",
        "is_correct": False,
    },
]

NUM_QUESTIONS = len(QUESTIONS)

# ── Session state ───────────────────────────────────────────────────────────
if "cal_question_idx" not in st.session_state:
    st.session_state.cal_question_idx = 0
if "cal_user_guesses" not in st.session_state:
    st.session_state.cal_user_guesses = []  # list of user's guesses ("Correct", "Incorrect", "Not Attempted")
if "cal_revealed" not in st.session_state:
    st.session_state.cal_revealed = False

# ── Tabs ────────────────────────────────────────────────────────────────────
section_tabs = st.tabs(["Calibration Quiz", "Calibration Curve Playground"])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Calibration Quiz
# ══════════════════════════════════════════════════════════════════════════════
with section_tabs[0]:
    current_idx = st.session_state.cal_question_idx

    # ── Final results screen ────────────────────────────────────────────────
    if current_idx >= NUM_QUESTIONS:
        st.markdown("### Quiz Complete!")
        st.markdown("---")

        # Calculate user accuracy at detecting wrong answers
        total_correct_detections = 0
        total_questions = NUM_QUESTIONS
        results_detail = []

        for i, q in enumerate(QUESTIONS):
            if i < len(st.session_state.cal_user_guesses):
                user_guess = st.session_state.cal_user_guesses[i]
                actual_label = "Correct" if q["is_correct"] else "Incorrect"
                detected = user_guess == actual_label
                if detected:
                    total_correct_detections += 1
                results_detail.append({
                    "question": q["question"],
                    "user_guess": user_guess,
                    "actual": actual_label,
                    "model_confidence": q["model_confidence"],
                    "detected": detected,
                })

        accuracy = total_correct_detections / total_questions if total_questions > 0 else 0

        st.markdown(f"### Your Accuracy: **{total_correct_detections} / {total_questions}** ({accuracy:.0%})")

        if accuracy == 1.0:
            st.success("Perfect score! You saw right through the model's confidence bluffs.")
        elif accuracy >= 0.8:
            st.success("Excellent! You have a strong sense of when to trust a model's stated confidence.")
        elif accuracy >= 0.6:
            st.warning("Good effort! Some hallucinations are harder to spot, especially when the model sounds confident.")
        else:
            st.error("Calibration is tricky! High confidence does not mean high accuracy. Review the results below.")

        # Results breakdown
        st.markdown("#### Question-by-Question Review")

        # Separate correct and incorrect model answers for analysis
        incorrect_qs = [r for r in results_detail if r["actual"] == "Incorrect"]
        correct_qs = [r for r in results_detail if r["actual"] == "Correct"]
        caught_hallucinations = sum(1 for r in incorrect_qs if r["detected"])
        false_alarms = sum(1 for r in correct_qs if not r["detected"])

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Hallucinations Caught", f"{caught_hallucinations}/{len(incorrect_qs)}")
        with col_m2:
            st.metric("False Alarms", f"{false_alarms}")
        with col_m3:
            avg_conf_wrong = np.mean([q["model_confidence"] for q in QUESTIONS if not q["is_correct"]])
            avg_conf_right = np.mean([q["model_confidence"] for q in QUESTIONS if q["is_correct"]])
            st.metric("Avg Confidence (Wrong)", f"{avg_conf_wrong:.0f}%")

        # Bar chart: model confidence vs actual correctness
        fig_review = go.Figure()

        q_labels = [f"Q{i+1}" for i in range(NUM_QUESTIONS)]
        confidences = [q["model_confidence"] for q in QUESTIONS]
        actual_correct = [1 if q["is_correct"] else 0 for q in QUESTIONS]
        bar_colors = [COLORS["green"] if q["is_correct"] else COLORS["red"] for q in QUESTIONS]

        fig_review.add_trace(go.Bar(
            x=q_labels,
            y=confidences,
            marker_color=bar_colors,
            opacity=0.85,
            name="Model Confidence",
            text=[f"{c}%" for c in confidences],
            textposition="outside",
        ))
        fig_review.add_hline(
            y=np.mean(confidences), line_dash="dot", line_color=COLORS["yellow"],
            annotation_text=f"Avg confidence: {np.mean(confidences):.0f}%",
            annotation_position="top right",
        )
        fig_review.update_layout(
            title="Model Confidence per Question (Green = Correct, Red = Incorrect)",
            yaxis_title="Stated Confidence (%)",
            yaxis=dict(range=[0, 110]),
            height=400,
        )
        st.plotly_chart(fig_review, use_container_width=True)

        # Detailed results
        for i, r in enumerate(results_detail):
            icon = "&#x2705;" if r["detected"] else "&#x274C;"
            q = QUESTIONS[i]
            st.markdown(f"""
<div class="concept-card">
{icon} <strong>Q{i+1}:</strong> {r['question']}<br/>
<span style="color:{COLORS['gray']};">Model said:</span> {q['model_answer']} ({q['model_confidence']}% confident)<br/>
<span style="color:{COLORS['gray']};">Truth:</span> {q['actual_answer']} &mdash;
<strong style="color:{COLORS['green'] if r['actual'] == 'Correct' else COLORS['red']};">{r['actual']}</strong><br/>
<span style="color:{COLORS['gray']};">You guessed:</span> <strong>{r['user_guess']}</strong>
{'&#x2705;' if r['detected'] else '&#x274C;'}
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="insight-box">
<strong>Key takeaway:</strong> The model's average confidence on <em>incorrect</em> answers was
<strong>{avg_conf_wrong:.0f}%</strong>, while on correct answers it was <strong>{avg_conf_right:.0f}%</strong>.
A well-calibrated model would show much lower confidence when it is wrong.
This confidence-accuracy gap is exactly what calibration measures.
</div>
""", unsafe_allow_html=True)

        if st.button("Play Again"):
            st.session_state.cal_question_idx = 0
            st.session_state.cal_user_guesses = []
            st.session_state.cal_revealed = False
            st.rerun()

    # ── Active question ─────────────────────────────────────────────────────
    else:
        q = QUESTIONS[current_idx]

        st.markdown(f"#### Question {current_idx + 1} of {NUM_QUESTIONS}")
        st.progress((current_idx) / NUM_QUESTIONS)

        # Display the question
        st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']};">Question:</strong><br/>
<span style="color:{COLORS['white']}; font-size:1.05rem;">{q['question']}</span>
</div>
""", unsafe_allow_html=True)

        # Display the model's answer and confidence
        conf_color = COLORS["green"] if q["model_confidence"] >= 90 else (
            COLORS["yellow"] if q["model_confidence"] >= 75 else COLORS["orange"]
        )
        st.markdown(f"""
<div class="concept-card" style="border-left: 4px solid {conf_color};">
<strong style="color:{COLORS['cyan']};">Model's Answer:</strong>
<span style="color:{COLORS['white']}; font-size:1.05rem;">{q['model_answer']}</span><br/><br/>
<strong style="color:{conf_color}; font-size:1.1rem;">{q['model_confidence']}% confident</strong>
</div>
""", unsafe_allow_html=True)

        # User hasn't guessed yet
        if not st.session_state.cal_revealed:
            st.markdown("**Do you think the model's answer is correct, incorrect, or not attempted?**")
            col_c, col_i, col_n = st.columns(3)
            with col_c:
                if st.button("Correct", key=f"cal_correct_{current_idx}", use_container_width=True):
                    st.session_state.cal_user_guesses.append("Correct")
                    st.session_state.cal_revealed = True
                    st.rerun()
            with col_i:
                if st.button("Incorrect", key=f"cal_incorrect_{current_idx}", use_container_width=True):
                    st.session_state.cal_user_guesses.append("Incorrect")
                    st.session_state.cal_revealed = True
                    st.rerun()
            with col_n:
                if st.button("Not Attempted", key=f"cal_na_{current_idx}", use_container_width=True):
                    st.session_state.cal_user_guesses.append("Not Attempted")
                    st.session_state.cal_revealed = True
                    st.rerun()

        # After guessing — reveal truth
        else:
            user_guess = st.session_state.cal_user_guesses[current_idx]
            actual_label = "Correct" if q["is_correct"] else "Incorrect"
            user_was_right = user_guess == actual_label

            if user_was_right:
                st.success(f"You got it! The model's answer is **{actual_label.lower()}**.")
            else:
                st.error(f"Not quite. The model's answer is actually **{actual_label.lower()}**. You guessed \"{user_guess}\".")

            if not q["is_correct"]:
                st.markdown(f"""
<div class="concept-card" style="border-left: 4px solid {COLORS['red']};">
<strong style="color:{COLORS['red']};">Ground Truth:</strong> {q['actual_answer']}<br/>
<span style="color:{COLORS['gray']};">The model said "{q['model_answer']}" with {q['model_confidence']}% confidence &mdash; a confident hallucination.</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="concept-card" style="border-left: 4px solid {COLORS['green']};">
<strong style="color:{COLORS['green']};">Confirmed Correct:</strong> {q['actual_answer']}<br/>
<span style="color:{COLORS['gray']};">The model's stated confidence of {q['model_confidence']}% was justified here.</span>
</div>
""", unsafe_allow_html=True)

            # Running score
            correct_so_far = sum(
                1 for i, g in enumerate(st.session_state.cal_user_guesses)
                if g == ("Correct" if QUESTIONS[i]["is_correct"] else "Incorrect")
            )
            st.markdown(f"**Your score so far:** {correct_so_far} / {current_idx + 1}")

            btn_label = "Next Question" if current_idx + 1 < NUM_QUESTIONS else "See Results"
            if st.button(f"{btn_label} \u2192"):
                st.session_state.cal_question_idx += 1
                st.session_state.cal_revealed = False
                st.rerun()

    # ── SimpleQA grading explanation ────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
<div class="insight-box">
<strong>SimpleQA Grading System:</strong> In the SimpleQA benchmark, model responses are graded into three categories:<br/><br/>
<strong style="color:{COLORS['green']};">Correct</strong> &mdash; The response fully contains the gold-standard target answer and does not contradict it.
The model may include additional context, but the core answer must be present and accurate.<br/><br/>
<strong style="color:{COLORS['red']};">Incorrect</strong> &mdash; The response directly contradicts the gold-standard target answer.
This is the hallucination case: the model states something factually wrong with confidence.<br/><br/>
<strong style="color:{COLORS['yellow']};">Not Attempted</strong> &mdash; The model declines to answer, hedges, or says it is unsure &mdash;
but crucially does <em>not</em> contradict the gold answer. This is calibrated behavior: the model knows what it does not know.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Calibration Curve Playground
# ══════════════════════════════════════════════════════════════════════════════
with section_tabs[1]:
    st.markdown("### Calibration Curve Playground")
    st.markdown(
        "Adjust the sliders to simulate a model with different accuracy and "
        "confidence characteristics. Watch how the calibration curve responds in real time."
    )

    # ── Sliders ─────────────────────────────────────────────────────────────
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        base_accuracy = st.slider(
            "Model Base Accuracy",
            min_value=0.3, max_value=0.8, value=0.5, step=0.05,
            help="The fraction of questions the model actually gets right.",
        )
    with col_s2:
        overconfidence = st.slider(
            "Overconfidence",
            min_value=-0.3, max_value=0.5, value=0.2, step=0.05,
            help="Positive values shift stated confidence above true accuracy. "
                 "Negative values make the model underconfident.",
        )
    with col_s3:
        conf_noise = st.slider(
            "Confidence Noise",
            min_value=0.0, max_value=0.3, value=0.1, step=0.02,
            help="Random noise added to each confidence score. "
                 "Higher values make confidence more scattered.",
        )

    # ── Explanation of what the sliders control ─────────────────────────────
    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['cyan']};">How the simulation works:</strong><br/>
We generate 100 simulated questions. Each is randomly correct or incorrect
based on the <strong>Base Accuracy</strong>. Then we assign each a stated confidence:<br/>
&bull; If correct: <code>conf = 0.5 + accuracy/2 + overconfidence + noise</code><br/>
&bull; If incorrect: <code>conf = 0.3 + overconfidence + noise</code><br/>
All values are clipped to [0, 1]. A perfectly calibrated model would have
its curve lie exactly on the diagonal.
</div>
""", unsafe_allow_html=True)

    # ── Generate simulated data ─────────────────────────────────────────────
    rng = np.random.RandomState(42)
    n_sim = 100

    is_correct = rng.rand(n_sim) < base_accuracy
    noise_vals = rng.randn(n_sim) * conf_noise

    stated_conf = np.where(
        is_correct,
        0.5 + base_accuracy / 2.0 + overconfidence + noise_vals,
        0.3 + overconfidence + noise_vals,
    )
    stated_conf = np.clip(stated_conf, 0.0, 1.0)

    # ── Bin into 10 calibration bins ────────────────────────────────────────
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(10):
        mask = (stated_conf >= bin_edges[i]) & (stated_conf < bin_edges[i + 1])
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            bin_accuracies.append(is_correct[mask].mean())
            bin_confidences.append(stated_conf[mask].mean())
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)

    # Filter out empty bins for plotting
    plot_centers = []
    plot_accuracies = []
    plot_confidences = []
    plot_counts = []
    for i in range(10):
        if bin_counts[i] > 0:
            plot_centers.append(bin_centers[i])
            plot_accuracies.append(bin_accuracies[i])
            plot_confidences.append(bin_confidences[i])
            plot_counts.append(bin_counts[i])

    plot_centers = np.array(plot_centers)
    plot_accuracies = np.array(plot_accuracies)
    plot_confidences = np.array(plot_confidences)
    plot_counts = np.array(plot_counts)

    # ── Calculate ECE ───────────────────────────────────────────────────────
    total_samples = sum(bin_counts)
    ece = 0.0
    for i in range(10):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_centers[i])

    # ── Calibration curve plot ──────────────────────────────────────────────
    fig_cal = go.Figure()

    # Fill area between curve and diagonal to visualize miscalibration
    if len(plot_centers) > 1:
        # Upper fill (above diagonal — overconfident regions)
        fig_cal.add_trace(go.Scatter(
            x=np.concatenate([plot_centers, plot_centers[::-1]]),
            y=np.concatenate([plot_accuracies, plot_centers[::-1]]),
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.2)",
            line=dict(width=0),
            name="Miscalibration Gap",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Perfect calibration diagonal
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=COLORS["gray"], width=2, dash="dash"),
        name="Perfect Calibration",
    ))

    # Actual calibration curve
    fig_cal.add_trace(go.Scatter(
        x=plot_centers,
        y=plot_accuracies,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=3),
        marker=dict(size=10, color=COLORS["blue"], line=dict(width=2, color=COLORS["white"])),
        name="Model Calibration",
        text=[f"Bin: {c:.0%}<br>Accuracy: {a:.0%}<br>Count: {n}"
              for c, a, n in zip(plot_centers, plot_accuracies, plot_counts)],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig_cal.update_layout(
        title="Calibration Curve: Stated Confidence vs. Actual Accuracy",
        xaxis_title="Stated Confidence",
        yaxis_title="Actual Accuracy (fraction correct in bin)",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=80),
    )

    st.plotly_chart(fig_cal, use_container_width=True)

    # ── ECE and metrics display ─────────────────────────────────────────────
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Expected Calibration Error (ECE)", f"{ece:.3f}")
    with col_e2:
        actual_acc = is_correct.mean()
        avg_stated = stated_conf.mean()
        st.metric("Actual Accuracy", f"{actual_acc:.0%}")
    with col_e3:
        st.metric("Avg Stated Confidence", f"{avg_stated:.0%}")

    # ── ECE formula ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="big-formula">
ECE = &sum;<sub>b=1</sub><sup>B</sup> (n<sub>b</sub> / N) &middot; |accuracy<sub>b</sub> &minus; confidence<sub>b</sub>|
</div>
""", unsafe_allow_html=True)

    # ── Confidence distribution histogram ───────────────────────────────────
    st.markdown("#### Confidence Distribution")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=stated_conf[is_correct],
        nbinsx=20,
        marker_color=COLORS["green"],
        opacity=0.7,
        name="Correct Answers",
    ))
    fig_hist.add_trace(go.Histogram(
        x=stated_conf[~is_correct],
        nbinsx=20,
        marker_color=COLORS["red"],
        opacity=0.7,
        name="Incorrect Answers",
    ))
    fig_hist.update_layout(
        barmode="overlay",
        title="Distribution of Stated Confidence by Correctness",
        xaxis_title="Stated Confidence",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=350,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Interpretation guide ────────────────────────────────────────────────
    if overconfidence > 0.1:
        interp_msg = (
            f"With overconfidence at <strong>{overconfidence:+.2f}</strong>, the model's curve "
            f"falls <em>below</em> the diagonal &mdash; it claims higher confidence than its accuracy justifies. "
            f"This is the most common failure mode: models that say \"95% sure\" but are right only 70% of the time."
        )
    elif overconfidence < -0.1:
        interp_msg = (
            f"With overconfidence at <strong>{overconfidence:+.2f}</strong>, the model's curve "
            f"rises <em>above</em> the diagonal &mdash; it is underconfident. "
            f"The model is more accurate than it claims, which is conservative but still miscalibrated."
        )
    else:
        interp_msg = (
            f"With overconfidence near zero (<strong>{overconfidence:+.2f}</strong>), "
            f"the model is close to well-calibrated. The curve approximately follows the diagonal."
        )

    st.markdown(f"""
<div class="insight-box">
<strong>Reading the curve:</strong> {interp_msg}<br/><br/>
<strong>ECE = {ece:.3f}</strong> &mdash; {"This is well-calibrated (ECE < 0.05)." if ece < 0.05 else "This indicates meaningful miscalibration." if ece < 0.15 else "This is severely miscalibrated. The model's confidence scores are unreliable."}
An ECE of 0 means perfect calibration; values above 0.1 indicate significant problems.
</div>
""", unsafe_allow_html=True)

    # ── Why calibration matters ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
<div class="insight-box">
<strong>Why calibration matters for AI safety:</strong>
A model that is overconfident in wrong answers is dangerous &mdash; users trust it when they should not.
The SimpleQA benchmark revealed that more capable models (like GPT-4) are often <em>less</em> calibrated
than smaller models: they hallucinate less frequently, but when they do, they are more confident about it.
The ideal behavior is not just being correct more often, but knowing when you do not know &mdash; and
saying "I'm not sure" instead of fabricating a confident answer.
</div>
""", unsafe_allow_html=True)
