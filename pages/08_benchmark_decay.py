import streamlit as st

st.set_page_config(page_title="Benchmark Decay", page_icon="📉", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">EVALS &amp; MODEL BEHAVIOR</p>', unsafe_allow_html=True)
st.title("Benchmark Decay")
st.markdown("Why do benchmarks stop working? Explore saturation, contamination, and format sensitivity.")
st.markdown("---")

# ── Session state init ───────────────────────────────────────────────────────
if "contam_answers" not in st.session_state:
    st.session_state.contam_answers = {}  # question index -> user guess ("clean" or "contaminated")
if "contam_submitted" not in st.session_state:
    st.session_state.contam_submitted = False

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Saturation Simulator", "Contamination Detective", "Format Sensitivity"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SATURATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Benchmark Saturation")
    st.markdown(
        "As models improve generation over generation, benchmark scores climb toward the ceiling. "
        "Once most models score above ~95%, the benchmark can no longer differentiate between them — it is **saturated**."
    )

    col_controls, col_chart = st.columns([1, 2])

    with col_controls:
        st.markdown("#### Parameters")
        difficulty = st.slider(
            "Benchmark difficulty",
            min_value=0.30, max_value=0.95, value=0.70, step=0.05,
            help="Starting accuracy of the weakest model. Lower = harder benchmark.",
            key="sat_difficulty",
        )
        n_generations = st.slider(
            "Model generations to simulate",
            min_value=5, max_value=20, value=10, step=1,
            help="Number of successive model releases to simulate.",
            key="sat_generations",
        )
        growth_rate = st.slider(
            "Capability growth rate",
            min_value=0.02, max_value=0.15, value=0.05, step=0.01,
            help="Score improvement per generation.",
            key="sat_growth",
        )

        st.markdown("#### Model Starting Offsets")
        st.markdown(f"""
<div class="concept-card">
<span style="color:{COLORS['blue']}">Weak model:</span> base = difficulty - 0.15<br/>
<span style="color:{COLORS['orange']}">Medium model:</span> base = difficulty<br/>
<span style="color:{COLORS['green']}">Strong model:</span> base = difficulty + 0.10
</div>
""", unsafe_allow_html=True)

    with col_chart:
        # Simulate three models
        rng = np.random.RandomState(42)
        generations = np.arange(n_generations)

        model_configs = [
            {"name": "Weak Model", "base": difficulty - 0.15, "color": COLORS["blue"]},
            {"name": "Medium Model", "base": difficulty, "color": COLORS["orange"]},
            {"name": "Strong Model", "base": difficulty + 0.10, "color": COLORS["green"]},
        ]

        fig_sat = go.Figure()

        all_scores = {}
        for cfg in model_configs:
            noise = rng.normal(0, 0.015, size=n_generations)
            scores = np.array([
                min(1.0, max(0.0, cfg["base"] + growth_rate * g + noise[g]))
                for g in generations
            ])
            all_scores[cfg["name"]] = scores

            fig_sat.add_trace(go.Scatter(
                x=generations,
                y=scores,
                mode="lines+markers",
                name=cfg["name"],
                line=dict(color=cfg["color"], width=2.5),
                marker=dict(size=6),
            ))

        # Saturation line
        fig_sat.add_hline(
            y=0.95, line_dash="dash", line_color=COLORS["red"],
            annotation_text="Saturation Line (0.95)",
            annotation_position="top left",
            annotation_font_color=COLORS["red"],
        )

        fig_sat.update_layout(
            title="Benchmark Score Over Model Generations",
            xaxis_title="Model Generation",
            yaxis_title="Benchmark Score",
            yaxis=dict(range=[0, 1.05]),
            height=480,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_sat, width="stretch")

    # Differentiation Power metric
    st.markdown("#### Differentiation Power")
    st.markdown(
        "How well can this benchmark tell models apart? Measured as the standard deviation of "
        "scores across the three models at each generation — when it drops near zero, the benchmark is dead."
    )

    diff_power = []
    for g in range(n_generations):
        scores_at_g = [all_scores[cfg["name"]][g] for cfg in model_configs]
        diff_power.append(np.std(scores_at_g))

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Bar(
        x=[f"Gen {g}" for g in generations],
        y=diff_power,
        marker_color=[
            COLORS["green"] if dp > 0.05 else (COLORS["orange"] if dp > 0.02 else COLORS["red"])
            for dp in diff_power
        ],
        opacity=0.85,
    ))
    fig_diff.update_layout(
        title="Differentiation Power (Score Spread Between Models)",
        xaxis_title="Model Generation",
        yaxis_title="Std Dev of Scores",
        height=320,
        yaxis=dict(range=[0, max(diff_power) * 1.3 + 0.01]),
    )
    st.plotly_chart(fig_diff, width="stretch")

    # Dead generation detection
    dead_gen = None
    for g in range(n_generations):
        scores_at_g = [all_scores[cfg["name"]][g] for cfg in model_configs]
        if all(s >= 0.95 for s in scores_at_g):
            dead_gen = g
            break

    if dead_gen is not None:
        st.warning(f"All three models cross 0.95 at generation {dead_gen}. The benchmark is effectively dead.")
    else:
        st.success("The benchmark still differentiates models across all simulated generations.")

    st.markdown(f"""
<div class="insight-box">
<strong>Real-world examples:</strong> GLUE saturated within a year of release (2018-2019).
SuperGLUE was designed to be harder but was surpassed by human performance by 2020.
MMLU, introduced in 2021, now sees top models scoring above 90%. Each time a benchmark
saturates, the community must create a harder one — a never-ending treadmill.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CONTAMINATION DETECTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Contamination Detective")
    st.markdown(
        "Some benchmark questions leak into training data — through web scrapes, GitHub repos, or forums. "
        "A contaminated model memorizes answers instead of reasoning. Can you spot which questions were leaked?"
    )

    # 8 benchmark questions: 4 clean, 4 contaminated
    QUESTIONS = [
        {
            "id": 0,
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
            "is_contaminated": False,
            "clean_score": 0.88,
            "contam_score": 0.85,
            "category": "Science",
        },
        {
            "id": 1,
            "question": "In the Winograd Schema: 'The trophy doesn't fit in the suitcase because it is too [large/small].' What does 'it' refer to?",
            "answer": "The trophy (if 'large') or the suitcase (if 'small')",
            "is_contaminated": True,
            "clean_score": 0.58,
            "contam_score": 0.97,
            "category": "NLU",
        },
        {
            "id": 2,
            "question": "Translate the following to French: 'The meeting has been rescheduled to next Thursday.'",
            "answer": "La reunion a ete reportee a jeudi prochain.",
            "is_contaminated": False,
            "clean_score": 0.82,
            "contam_score": 0.80,
            "category": "Translation",
        },
        {
            "id": 3,
            "question": "GSM8K #7241: A store sells notebooks for $4 each. Maria buys 3 notebooks and pays with a $20 bill. How much change does she receive?",
            "answer": "$8",
            "is_contaminated": True,
            "clean_score": 0.62,
            "contam_score": 0.99,
            "category": "Math",
        },
        {
            "id": 4,
            "question": "What literary device is used in 'The wind whispered through the trees'?",
            "answer": "Personification",
            "is_contaminated": False,
            "clean_score": 0.75,
            "contam_score": 0.73,
            "category": "Language Arts",
        },
        {
            "id": 5,
            "question": "MMLU Question ID 44201: In monetary policy, the Taylor Rule relates the nominal interest rate to: (A) inflation only (B) inflation and output gap (C) output gap only (D) exchange rates",
            "answer": "(B) inflation and output gap",
            "is_contaminated": True,
            "clean_score": 0.55,
            "contam_score": 0.96,
            "category": "Economics",
        },
        {
            "id": 6,
            "question": "Explain one advantage of renewable energy sources over fossil fuels.",
            "answer": "Renewable energy produces little to no greenhouse gas emissions during operation.",
            "is_contaminated": False,
            "clean_score": 0.90,
            "contam_score": 0.88,
            "category": "Environment",
        },
        {
            "id": 7,
            "question": "HumanEval/042: Implement a function that increments a list of digits representing a number. E.g., [1,2,3] -> [1,2,4], [9,9] -> [1,0,0].",
            "answer": "def increment(digits): carry = 1; ...",
            "is_contaminated": True,
            "clean_score": 0.60,
            "contam_score": 0.98,
            "category": "Code",
        },
    ]

    # Quiz UI
    if not st.session_state.contam_submitted:
        st.markdown(
            "For each question below, decide: was it **clean** (the model must reason from scratch) "
            "or **contaminated** (it appeared in training data, so the model memorized the answer)?"
        )
        st.markdown("")

        for q in QUESTIONS:
            qid = q["id"]
            st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['cyan']}">Q{qid + 1} [{q['category']}]:</strong>
<span style="color:{COLORS['white']}">{q['question']}</span>
</div>
""", unsafe_allow_html=True)
            user_guess = st.radio(
                f"Your verdict for Q{qid + 1}:",
                options=["Clean", "Contaminated"],
                key=f"contam_guess_{qid}",
                horizontal=True,
            )
            st.session_state.contam_answers[qid] = user_guess.lower()

        st.markdown("")
        if st.button("Submit Answers", key="contam_submit"):
            st.session_state.contam_submitted = True
            st.rerun()

    else:
        # Results
        correct_count = 0
        for q in QUESTIONS:
            qid = q["id"]
            truth = "contaminated" if q["is_contaminated"] else "clean"
            user_guess = st.session_state.contam_answers.get(qid, "clean")
            is_correct = user_guess == truth
            if is_correct:
                correct_count += 1

            icon = "&#x2705;" if is_correct else "&#x274C;"
            truth_label = "Contaminated" if q["is_contaminated"] else "Clean"
            guess_label = user_guess.capitalize()

            st.markdown(f"""
<div class="concept-card">
{icon} <strong style="color:{COLORS['cyan']}">Q{qid + 1} [{q['category']}]:</strong>
<span style="color:{COLORS['white']}">{q['question']}</span><br/>
<span style="color:{COLORS['gray']}">Your guess: <strong>{guess_label}</strong> | Actual: <strong style="color:{COLORS['green'] if is_correct else COLORS['red']}">{truth_label}</strong></span>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"### You got **{correct_count} / {len(QUESTIONS)}** correct!")

        if correct_count == len(QUESTIONS):
            st.success("Perfect score! You have a keen eye for contamination signals.")
        elif correct_count >= 6:
            st.success("Great job! You caught most contamination patterns.")
        elif correct_count >= 4:
            st.warning("Decent effort — contamination can be subtle. Check the clues below.")
        else:
            st.error("Contamination is tricky to spot! Review the signals below.")

        # Bar chart: Clean model vs Contaminated model
        st.markdown("#### Clean Training vs. Contaminated Training")
        st.markdown(
            "Below: **Model A** was trained cleanly (no benchmark leakage). "
            "**Model B** had these benchmark questions in its training data. "
            "Notice how Model B scores suspiciously high on contaminated questions."
        )

        q_labels = [f"Q{q['id']+1}\n{q['category']}" for q in QUESTIONS]
        clean_scores = [q["clean_score"] for q in QUESTIONS]
        contam_scores = [q["contam_score"] for q in QUESTIONS]
        is_contam = [q["is_contaminated"] for q in QUESTIONS]

        # Color contaminated bars differently
        contam_bar_colors = [
            COLORS["red"] if ic else COLORS["orange"]
            for ic in is_contam
        ]

        fig_contam = go.Figure()
        fig_contam.add_trace(go.Bar(
            x=q_labels,
            y=clean_scores,
            name="Model A (Clean Training)",
            marker_color=COLORS["blue"],
            opacity=0.85,
        ))
        fig_contam.add_trace(go.Bar(
            x=q_labels,
            y=contam_scores,
            name="Model B (Contaminated Training)",
            marker_color=contam_bar_colors,
            opacity=0.85,
        ))
        fig_contam.update_layout(
            barmode="group",
            title="Per-Question Accuracy: Clean vs. Contaminated Model",
            xaxis_title="Question",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.1]),
            height=450,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(b=100),
        )

        # Add annotation for contaminated questions
        for i, q in enumerate(QUESTIONS):
            if q["is_contaminated"]:
                fig_contam.add_annotation(
                    x=q_labels[i],
                    y=q["contam_score"] + 0.04,
                    text="LEAKED",
                    showarrow=False,
                    font=dict(color=COLORS["red"], size=10, family="Inter, sans-serif"),
                )

        st.plotly_chart(fig_contam, width="stretch")

        # Score comparison
        clean_avg_clean = np.mean([q["clean_score"] for q in QUESTIONS if not q["is_contaminated"]])
        clean_avg_contam = np.mean([q["clean_score"] for q in QUESTIONS if q["is_contaminated"]])
        dirty_avg_clean = np.mean([q["contam_score"] for q in QUESTIONS if not q["is_contaminated"]])
        dirty_avg_contam = np.mean([q["contam_score"] for q in QUESTIONS if q["is_contaminated"]])

        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<strong style="color:{COLORS['blue']}">Model A (Clean)</strong><br/><br/>
On clean questions: <strong>{clean_avg_clean:.0%}</strong><br/>
On contaminated questions: <strong>{clean_avg_contam:.0%}</strong><br/>
<span style="color:{COLORS['gray']}">Gap: {abs(clean_avg_clean - clean_avg_contam):.0%}</span>
</div>
""", unsafe_allow_html=True)
        with col_stats2:
            st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<strong style="color:{COLORS['red']}">Model B (Contaminated)</strong><br/><br/>
On clean questions: <strong>{dirty_avg_clean:.0%}</strong><br/>
On contaminated questions: <strong>{dirty_avg_contam:.0%}</strong><br/>
<span style="color:{COLORS['red']}">Gap: {abs(dirty_avg_clean - dirty_avg_contam):.0%}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="insight-box">
<strong>How to spot contamination:</strong> Look for question IDs, dataset names (GSM8K, MMLU,
HumanEval), or suspiciously specific numbering — these are signs a question was copied verbatim
from a published benchmark. Real-world defenses include <strong>canary strings</strong>
(unique identifiers embedded in test sets), <strong>held-out private test splits</strong>, and
<strong>dynamic benchmark generation</strong>. The Llama 4 benchmarks controversy showed how
teams can optimize specifically for known evaluation sets, inflating scores without genuine
capability gains.
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        if st.button("Try Again", key="contam_reset"):
            st.session_state.contam_answers = {}
            st.session_state.contam_submitted = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: FORMAT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Format Sensitivity")
    st.markdown(
        "The **same** model on the **same** question can produce wildly different benchmark scores "
        "depending on trivial formatting choices in the prompt template. "
        "This means benchmark results are partly measuring prompt formatting, not model capability."
    )

    # The benchmark question
    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Benchmark Task:</strong> Reading Comprehension<br/><br/>
<strong style="color:{COLORS['cyan']}">Passage:</strong>
<span style="color:{COLORS['white']}">
The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest
coral reef system. It is composed of over 2,900 individual reef systems and hundreds of islands.
The reef supports a wide diversity of life and was selected as a World Heritage Site in 1981.
</span><br/><br/>
<strong style="color:{COLORS['cyan']}">Question:</strong>
<span style="color:{COLORS['white']}">Where is the Great Barrier Reef located?</span><br/>
<strong style="color:{COLORS['green']}">Correct Answer:</strong>
<span style="color:{COLORS['white']}">Off the coast of Queensland, Australia</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### How does the prompt template affect accuracy?")
    st.markdown(
        "Below are five ways to format the exact same question for the model. "
        "Select one to highlight, then compare all five in the chart."
    )

    FORMAT_VARIANTS = [
        {
            "name": 'Passage: <text>\\nAnswer: <text>',
            "template": 'Passage: The Great Barrier Reef, located off the coast...\nQuestion: Where is the Great Barrier Reef located?\nAnswer:',
            "score": 0.80,
            "label": "Standard format",
        },
        {
            "name": 'Passage:<text>\\nAnswer:<text>  (no space)',
            "template": 'Passage:The Great Barrier Reef, located off the coast...\nQuestion:Where is the Great Barrier Reef located?\nAnswer:',
            "score": 0.45,
            "label": "Missing spaces after colons",
        },
        {
            "name": 'PASSAGE: <text>\\nANSWER: <text>  (caps)',
            "template": 'PASSAGE: The Great Barrier Reef, located off the coast...\nQUESTION: Where is the Great Barrier Reef located?\nANSWER:',
            "score": 0.62,
            "label": "Uppercase keywords",
        },
        {
            "name": 'Passage <text> Answer <text>  (no colon)',
            "template": 'Passage The Great Barrier Reef, located off the coast...\nQuestion Where is the Great Barrier Reef located?\nAnswer',
            "score": 0.35,
            "label": "No colons at all",
        },
        {
            "name": 'passage: <text> answer: <text>  (lowercase)',
            "template": 'passage: The Great Barrier Reef, located off the coast...\nquestion: Where is the Great Barrier Reef located?\nanswer:',
            "score": 0.52,
            "label": "All lowercase keywords",
        },
    ]

    selected_format = st.radio(
        "Select a prompt format to inspect:",
        options=[f["name"] for f in FORMAT_VARIANTS],
        key="format_select",
    )

    selected_idx = next(i for i, f in enumerate(FORMAT_VARIANTS) if f["name"] == selected_format)
    selected_variant = FORMAT_VARIANTS[selected_idx]

    col_template, col_score = st.columns([2, 1])
    with col_template:
        st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['orange']}">Prompt Template:</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {selected_variant['label']}</span>
<br/><br/>
<div class="big-formula" style="white-space: pre-wrap; font-size: 0.95rem;">{selected_variant['template']}</div>
</div>
""", unsafe_allow_html=True)
    with col_score:
        score_color = (
            COLORS["green"] if selected_variant["score"] >= 0.70
            else (COLORS["orange"] if selected_variant["score"] >= 0.50 else COLORS["red"])
        )
        st.markdown(f"""
<div class="concept-card" style="text-align:center; padding: 30px 20px;">
<span style="color:{COLORS['gray']}; font-size:0.9rem;">Model Accuracy</span><br/>
<span style="color:{score_color}; font-size:3rem; font-weight:700;">{selected_variant['score']:.0%}</span><br/>
<span style="color:{COLORS['gray']}; font-size:0.8rem;">on this format</span>
</div>
""", unsafe_allow_html=True)

    # Bar chart of all formats
    st.markdown("#### All Formats Compared")

    format_names = [f["label"] for f in FORMAT_VARIANTS]
    format_scores = [f["score"] for f in FORMAT_VARIANTS]

    bar_colors = []
    for i, f in enumerate(FORMAT_VARIANTS):
        if i == selected_idx:
            bar_colors.append(COLORS["cyan"])
        elif f["score"] >= 0.70:
            bar_colors.append(COLORS["green"])
        elif f["score"] >= 0.50:
            bar_colors.append(COLORS["orange"])
        else:
            bar_colors.append(COLORS["red"])

    fig_format = go.Figure()
    fig_format.add_trace(go.Bar(
        x=format_names,
        y=format_scores,
        marker_color=bar_colors,
        opacity=0.9,
        text=[f"{s:.0%}" for s in format_scores],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=13),
    ))

    # Add range annotation
    max_score = max(format_scores)
    min_score = min(format_scores)
    spread = max_score - min_score

    fig_format.add_annotation(
        x=format_names[np.argmax(format_scores)],
        y=max_score + 0.08,
        text=f"Spread: {spread:.0%}",
        showarrow=False,
        font=dict(color=COLORS["red"], size=14, family="Inter, sans-serif"),
    )

    fig_format.update_layout(
        title="Same Model, Same Question — Different Formats",
        xaxis_title="Prompt Format",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.15]),
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_format, width="stretch")

    # Spread metric
    st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="color:{COLORS['gray']}">Accuracy Range</span><br/>
<span style="color:{COLORS['white']}; font-size:1.8rem; font-weight:700;">
{min_score:.0%} &mdash; {max_score:.0%}
</span><br/>
<span style="color:{COLORS['red']}; font-size:1.1rem;">
Spread: {spread:.0%}
</span><br/><br/>
<span style="color:{COLORS['gray']}; font-size:0.85rem;">
A single space or colon changes accuracy by up to {spread:.0%}.
How meaningful is a "benchmark score" when it shifts this much from formatting alone?
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="insight-box">
<strong>Why this matters:</strong> Research has shown accuracy spreads as wide as 0.036 to 0.804
on the same task depending purely on prompt format. This means when a paper reports
"our model achieves 80% on BenchmarkX," the format template is doing significant work.
Responsible evaluation requires reporting results across <strong>multiple prompt templates</strong>,
using <strong>standardized evaluation harnesses</strong> (like LM Evaluation Harness or HELM),
and always disclosing the exact prompt format used.
</div>
""", unsafe_allow_html=True)
