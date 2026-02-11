import streamlit as st

st.set_page_config(page_title="Annotator Disagreement", page_icon="🗳️", layout="wide")

from style import inject_custom_css, COLORS, sigmoid
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

st.markdown('<p class="section-header">HUMAN PREFERENCE NOISE</p>', unsafe_allow_html=True)
st.title("Annotator Disagreement Simulator")
st.markdown("Human preferences aren't ground truth — they're **noisy**, **plural**, and sometimes **irreconcilable**. This limits how well any reward model can learn.")
st.markdown("---")

# ── Annotator personas ───────────────────────────────────────────────────────
ANNOTATORS = [
    {"name": "Safety Sam", "icon": "🛡️", "bias": "safety",
     "desc": "Prioritizes harm avoidance above all. Prefers cautious, hedged responses."},
    {"name": "Helpful Hana", "icon": "💡", "bias": "helpfulness",
     "desc": "Values directness and practical utility. Dislikes unnecessary caveats."},
    {"name": "Creative Carlos", "icon": "🎨", "bias": "creativity",
     "desc": "Rewards originality, wit, and engaging writing style."},
    {"name": "Accuracy Aisha", "icon": "🔬", "bias": "accuracy",
     "desc": "Cares only about factual correctness and precision."},
    {"name": "Empathy Eli", "icon": "💚", "bias": "empathy",
     "desc": "Values emotional intelligence, warmth, and user understanding."},
]

# ── Prompts for Phase 1 ─────────────────────────────────────────────────────
PROMPTS = [
    {
        "id": "diet",
        "prompt": "I want to lose weight fast. What's the best approach?",
        "response_a": "A calorie deficit of 500-750 cal/day through balanced nutrition and exercise is sustainable and safe. Aim for 1-2 lbs/week — faster loss risks muscle loss and metabolic adaptation.",
        "response_b": "Try intermittent fasting with a 16:8 window, cut carbs significantly, and do HIIT cardio 5x/week. You could lose 5-10 lbs in the first week, though some will be water weight.",
        "label_a": "Cautious & evidence-based",
        "label_b": "Direct & actionable",
        # Annotator votes: True = prefers A, False = prefers B
        "annotator_votes": [True, False, False, True, True],
        "dimension_scores": {
            "safety": [0.9, 0.3], "helpfulness": [0.5, 0.9],
            "creativity": [0.3, 0.6], "accuracy": [0.8, 0.6],
            "empathy": [0.7, 0.5],
        },
    },
    {
        "id": "code",
        "prompt": "Write me a Python function to check if a number is prime.",
        "response_a": "def is_prime(n): if n &lt; 2: return False; for i in range(2, int(n**0.5) + 1): if n % i == 0: return False; return True",
        "response_b": "Here's a prime checker! I'll walk you through the logic step by step so you understand why it works, not just how. A prime number is only divisible by 1 and itself. So we check all possible divisors up to \u221An (we only need to go to the square root \u2014 can you see why?). The function checks if n &lt; 2, then loops from 2 to \u221An, returning False if any divisor is found.",
        "label_a": "Concise code only",
        "label_b": "Explanatory walkthrough",
        "annotator_votes": [True, False, False, True, False],
        "dimension_scores": {
            "safety": [0.7, 0.7], "helpfulness": [0.6, 0.8],
            "creativity": [0.3, 0.7], "accuracy": [0.9, 0.9],
            "empathy": [0.3, 0.8],
        },
    },
    {
        "id": "joke",
        "prompt": "Tell me a joke about programming.",
        "response_a": "Why do programmers prefer dark mode? Because light attracts bugs.",
        "response_b": "I'd be happy to share a programming joke! Here's one: There are only 10 types of people in the world — those who understand binary, and those who don't. I hope that gave you a smile! Let me know if you'd like to hear more.",
        "label_a": "Quick & punchy",
        "label_b": "Friendly but padded",
        "annotator_votes": [True, False, True, True, False],
        "dimension_scores": {
            "safety": [0.8, 0.9], "helpfulness": [0.7, 0.5],
            "creativity": [0.8, 0.5], "accuracy": [0.5, 0.5],
            "empathy": [0.5, 0.7],
        },
    },
]

# ── Hard cases for Phase 3 ──────────────────────────────────────────────────
HARD_CASES = [
    {
        "prompt": "My friend told me a secret and asked me not to tell anyone, but the secret is that they're being bullied at school. Should I tell an adult?",
        "response_a": "Yes, you should tell a trusted adult. Your friend's safety is more important than keeping the secret. Bullying can escalate, and adults can intervene in ways you can't. Your friend may be upset at first, but you'd be helping them.",
        "response_b": "This is a tough situation. Before telling an adult, try talking to your friend first — let them know you're worried and encourage *them* to seek help. Breaking their trust could damage your friendship and make them less likely to confide in anyone in the future.",
        "label_a": "Prioritize safety — tell an adult",
        "label_b": "Prioritize trust — talk to friend first",
        "annotator_votes": [True, False, False, True, True],
        "split": "3-2 for A",
    },
    {
        "prompt": "Should AI systems be allowed to generate creative fiction that includes morally complex characters who do bad things?",
        "response_a": "Yes — great literature has always explored moral complexity. Characters like Macbeth, Walter White, and Humbert Humbert teach us about human nature precisely because they are flawed. AI should be able to create similarly complex narratives.",
        "response_b": "AI-generated content reaches a massive audience without the cultural context that surrounds published literature. There's a real risk of normalizing harmful behavior, especially for younger users who may not critically evaluate what they read. Caution is warranted.",
        "label_a": "Allow it — art requires moral complexity",
        "label_b": "Restrict it — risk of normalization",
        "annotator_votes": [False, True, True, False, False],
        "split": "3-2 for B",
    },
    {
        "prompt": "A user asks an AI assistant to help write a strongly-worded complaint letter about a neighbor. The letter is aggressive but not threatening. Should the AI help?",
        "response_a": "Yes, the AI should help. The user has a right to express frustration, and writing a letter is a legitimate, non-violent way to address a conflict. The AI can help while suggesting constructive framing.",
        "response_b": "The AI should suggest a more measured tone. Aggressive language, even if not threatening, can escalate conflicts. The AI's role should include nudging users toward more productive communication.",
        "label_a": "Help as requested — user autonomy",
        "label_b": "Suggest softer tone — reduce harm",
        "annotator_votes": [False, True, True, False, False],
        "split": "3-2 for B",
    },
]

# ── Session state ────────────────────────────────────────────────────────────
if "annot_votes" not in st.session_state:
    st.session_state.annot_votes = {}  # prompt_id -> "A" or "B"

phase_tabs = st.tabs(["Phase 1: Vote & Compare", "Phase 2: Noise Destroys Accuracy", "Phase 3: Hard Cases"])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Student votes, then sees annotator disagreement
# ══════════════════════════════════════════════════════════════════════════════
with phase_tabs[0]:
    st.markdown("### Your Vote vs. The Annotators")
    st.markdown("For each prompt, pick the response you prefer. Then see how 5 annotators with different priorities voted.")

    all_voted = True
    for p in PROMPTS:
        pid = p["id"]
        st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Prompt:</strong> {p['prompt']}
</div>
""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
<div class="concept-card" style="min-height:100px;">
<strong style="color:{COLORS['cyan']}">Response A</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {p['label_a']}</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{p['response_a']}</span>
</div>
""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
<div class="concept-card" style="min-height:100px;">
<strong style="color:{COLORS['orange']}">Response B</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {p['label_b']}</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{p['response_b']}</span>
</div>
""", unsafe_allow_html=True)

        if pid not in st.session_state.annot_votes:
            all_voted = False
            col_btn_a, col_btn_b = st.columns(2)
            with col_btn_a:
                if st.button(f"Prefer A", key=f"vote_a_{pid}"):
                    st.session_state.annot_votes[pid] = "A"
                    st.rerun()
            with col_btn_b:
                if st.button(f"Prefer B", key=f"vote_b_{pid}"):
                    st.session_state.annot_votes[pid] = "B"
                    st.rerun()
        else:
            user_vote = st.session_state.annot_votes[pid]
            st.markdown(f"**Your vote: Response {user_vote}**")

        st.markdown("")

    # Show agreement heatmap once all voted
    if all_voted:
        st.markdown("---")
        st.markdown("### Agreement Heatmap")
        st.markdown("Each cell shows whether the annotator (or you) preferred **A** or **B**.")

        # Build heatmap data: rows = prompts, cols = [You] + annotators
        col_names = ["You"] + [a["icon"] + " " + a["name"] for a in ANNOTATORS]
        row_names = [p["prompt"][:50] + "..." for p in PROMPTS]

        z_data = []
        text_data = []
        for p in PROMPTS:
            row_z = []
            row_text = []
            # User vote
            user_a = st.session_state.annot_votes[p["id"]] == "A"
            row_z.append(1 if user_a else 0)
            row_text.append("A" if user_a else "B")
            # Annotator votes
            for vote in p["annotator_votes"]:
                row_z.append(1 if vote else 0)
                row_text.append("A" if vote else "B")
            z_data.append(row_z)
            text_data.append(row_text)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data,
            x=col_names,
            y=row_names,
            text=text_data,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            colorscale=[[0, COLORS["orange"]], [1, COLORS["cyan"]]],
            showscale=False,
            hovertemplate="Voter: %{x}<br>Prompt: %{y}<br>Vote: %{text}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Who Agrees With Whom?",
            height=300,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_heat, width="stretch")

        # Agreement stats
        st.markdown("#### Annotator Profiles")
        annot_cols = st.columns(5)
        for i, ann in enumerate(ANNOTATORS):
            with annot_cols[i]:
                agreements = sum(
                    1 for p in PROMPTS
                    if (p["annotator_votes"][i] and st.session_state.annot_votes[p["id"]] == "A")
                    or (not p["annotator_votes"][i] and st.session_state.annot_votes[p["id"]] == "B")
                )
                st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:2rem;">{ann['icon']}</span><br/>
<strong style="color:{COLORS['cyan']}">{ann['name']}</strong><br/>
<span style="color:{COLORS['gray']}; font-size:0.8rem;">{ann['desc']}</span><br/><br/>
<strong>Agreement with you: {agreements}/{len(PROMPTS)}</strong>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="insight-box">
<strong>Key takeaway:</strong> Even on these straightforward prompts, annotators disagree.
When we train a reward model on majority-vote labels, we're throwing away the minority
perspective — but the minority may have valid reasons for their preference.
</div>
""", unsafe_allow_html=True)

    else:
        remaining = len(PROMPTS) - len(st.session_state.annot_votes)
        st.info(f"Vote on {remaining} more prompt(s) to see the agreement heatmap.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Disagreement slider → RM accuracy
# ══════════════════════════════════════════════════════════════════════════════
with phase_tabs[1]:
    st.markdown("### How Noise Destroys Reward Model Accuracy")
    st.markdown("If annotators disagree, the labels we train on are noisy. How does this affect the RM?")

    noise_level = st.slider("Annotator disagreement rate", 0.0, 0.5, 0.15, 0.05,
                             help="Probability that each annotator's label is flipped from 'true' preference.")

    rng = np.random.RandomState(123)
    n_pairs = 500
    true_margins = rng.randn(n_pairs)  # true quality difference (A - B)

    # Simulate noisy labels: each of 5 annotators votes, with noise
    n_annotators = 5
    correct_probs = sigmoid(true_margins)  # P(annotator picks A) based on true margin

    # Sweep noise levels
    noise_sweep = np.linspace(0, 0.5, 50)
    acc_1ann = []
    acc_3ann = []
    acc_5ann = []

    for nl in noise_sweep:
        for n_ann, acc_list in [(1, acc_1ann), (3, acc_3ann), (5, acc_5ann)]:
            rng2 = np.random.RandomState(42)
            correct = 0
            for i in range(n_pairs):
                votes = []
                for _ in range(n_ann):
                    if rng2.rand() < nl:
                        vote = rng2.rand() > correct_probs[i]
                    else:
                        vote = rng2.rand() < correct_probs[i]
                    votes.append(vote)
                majority_a = sum(votes) > n_ann / 2
                true_a = true_margins[i] > 0
                if majority_a == true_a:
                    correct += 1
            acc_list.append(correct / n_pairs)

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=noise_sweep, y=acc_1ann,
        mode="lines", name="1 annotator",
        line=dict(color=COLORS["red"], width=2),
    ))
    fig_acc.add_trace(go.Scatter(
        x=noise_sweep, y=acc_3ann,
        mode="lines", name="3 annotators (majority)",
        line=dict(color=COLORS["orange"], width=2),
    ))
    fig_acc.add_trace(go.Scatter(
        x=noise_sweep, y=acc_5ann,
        mode="lines", name="5 annotators (majority)",
        line=dict(color=COLORS["green"], width=2),
    ))

    # Current noise level marker
    fig_acc.add_vline(x=noise_level, line_dash="dot", line_color=COLORS["cyan"],
                       annotation_text=f"Current: {noise_level:.0%}",
                       annotation_position="bottom right")

    fig_acc.update_layout(
        xaxis_title="Annotator Disagreement Rate",
        yaxis_title="Label Accuracy (majority vote)",
        title="More Annotators Help — But Can't Fix Fundamental Disagreement",
        height=450,
        yaxis=dict(range=[0.45, 1.0]),
        xaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_acc, width="stretch")

    # Confidence distribution
    st.markdown("#### Reward Model Confidence Distribution")
    st.markdown("How confident should the RM be, given this level of annotator noise?")

    rng3 = np.random.RandomState(77)
    rm_scores = []
    for i in range(n_pairs):
        votes = []
        for _ in range(5):
            if rng3.rand() < noise_level:
                vote = rng3.rand() > correct_probs[i]
            else:
                vote = rng3.rand() < correct_probs[i]
            votes.append(vote)
        # RM "confidence" = fraction voting A
        rm_scores.append(sum(votes) / 5)

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=rm_scores,
        nbinsx=20,
        marker_color=COLORS["blue"],
        opacity=0.8,
        name="RM Confidence",
    ))
    fig_conf.update_layout(
        xaxis_title="Fraction of Annotators Preferring A",
        yaxis_title="Number of Pairs",
        title="Distribution of Annotator Agreement",
        height=350,
    )
    st.plotly_chart(fig_conf, width="stretch")

    st.markdown(f"""
<div class="insight-box">
<strong>Implication:</strong> At {noise_level:.0%} disagreement, many comparisons have near-50/50
splits. The RM is forced to learn confident scores from ambiguous data. This is a fundamental
limit on alignment quality — not a fixable engineering problem.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Hard cases — genuinely ambiguous
# ══════════════════════════════════════════════════════════════════════════════
with phase_tabs[2]:
    st.markdown("### Hard Cases: No Right Answer")
    st.markdown("These cases split our annotators 3-2. There is no consensus. What would *you* choose — and what should the RM learn?")

    for idx, hc in enumerate(HARD_CASES):
        st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Prompt:</strong> {hc['prompt']}
</div>
""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
<div class="concept-card" style="min-height:100px;">
<strong style="color:{COLORS['cyan']}">Response A</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {hc['label_a']}</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{hc['response_a']}</span>
</div>
""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
<div class="concept-card" style="min-height:100px;">
<strong style="color:{COLORS['orange']}">Response B</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {hc['label_b']}</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{hc['response_b']}</span>
</div>
""", unsafe_allow_html=True)

        # Show annotator votes
        st.markdown(f"**Annotator split: {hc['split']}**")
        vote_cols = st.columns(5)
        for i, ann in enumerate(ANNOTATORS):
            with vote_cols[i]:
                vote = "A" if hc["annotator_votes"][i] else "B"
                vote_color = COLORS["cyan"] if vote == "A" else COLORS["orange"]
                st.markdown(f"""
<div style="text-align:center; padding:8px;">
<span style="font-size:1.5rem;">{ann['icon']}</span><br/>
<strong style="color:{vote_color}">{vote}</strong>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

    st.markdown(f"""
<div class="insight-box">
<strong>The deeper problem:</strong> These aren't cases of annotator error — they reflect genuinely
different values. Safety vs. helpfulness, autonomy vs. paternalism, artistic freedom vs. harm reduction.
A single reward model <em>cannot</em> represent all these perspectives simultaneously.
This is why research on <strong>pluralistic alignment</strong> and <strong>constitutional AI</strong> matters.
</div>
""", unsafe_allow_html=True)

    # Reset button
    st.markdown("")
    if st.button("Reset All Votes"):
        st.session_state.annot_votes = {}
        st.rerun()
