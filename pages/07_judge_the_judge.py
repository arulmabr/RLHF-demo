import streamlit as st

st.set_page_config(page_title="Judge the Judge", page_icon="⚖️", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Matchup data: 6 prompt+response pairs ──────────────────────────────────
MATCHUPS = [
    {
        "id": "quantum",
        "prompt": "Explain quantum computing to a 10-year-old",
        "bias_tested": "Length Bias",
        "response_a": {
            "text": "Imagine you have a magic coin. A normal coin is either heads or tails. But a quantum coin can be both at the same time until you look at it! A quantum computer uses tiny particles that work like magic coins. Because they can be many things at once, they can try lots of answers to a problem all at the same time, instead of one by one like a regular computer.",
            "length": 55,
            "has_formatting": False,
            "true_quality": 0.88,
        },
        "response_b": {
            "text": "Quantum computing is a type of computation that utilizes quantum mechanical phenomena. Let me break this down for you in detail.\n\nFirst, classical computers use bits, which are either 0 or 1. Quantum computers use quantum bits, or qubits, which can exist in a superposition of states. This means they can represent both 0 and 1 simultaneously due to the principle of superposition.\n\nSecond, there is a phenomenon called entanglement, where two qubits become correlated. When you measure one, you instantly know the state of the other, regardless of distance. Einstein called this 'spooky action at a distance.'\n\nThird, quantum interference allows quantum computers to amplify correct answers and cancel wrong ones. This gives them an advantage for certain types of problems.\n\nIn summary, quantum computers leverage superposition, entanglement, and interference to solve specific classes of problems faster than classical computers. However, they are not universally faster and face challenges like decoherence and error correction. Current quantum computers have limited qubits and are still largely experimental.",
            "length": 190,
            "has_formatting": False,
            "true_quality": 0.52,
        },
    },
    {
        "id": "sorting",
        "prompt": "What is the most efficient sorting algorithm?",
        "bias_tested": "Format Bias",
        "response_a": {
            "text": "It depends on the situation. For general-purpose sorting, merge sort and heapsort guarantee O(n log n) worst-case performance. In practice, Timsort (used by Python and Java) is often fastest because it exploits existing order in real-world data. For small arrays, insertion sort wins due to low overhead. If you know your data is integers in a bounded range, radix sort runs in linear time. There is no single 'best' algorithm without knowing your constraints.",
            "length": 65,
            "has_formatting": False,
            "true_quality": 0.92,
        },
        "response_b": {
            "text": "# Sorting Algorithms Compared\n\n## Top Algorithms\n| Algorithm | Time (Avg) | Time (Worst) | Space |\n|-----------|-----------|-------------|-------|\n| **QuickSort** | O(n log n) | O(n^2) | O(log n) |\n| **MergeSort** | O(n log n) | O(n log n) | O(n) |\n| **HeapSort** | O(n log n) | O(n log n) | O(1) |\n\n## Key Takeaway\n> The **best** sorting algorithm is **QuickSort** for most cases!\n\n## Why QuickSort Wins\n- Fast in practice due to cache locality\n- In-place sorting (low memory)\n- Average case is O(n log n)\n\n---\n*Note: Always consider your specific use case!*",
            "length": 120,
            "has_formatting": True,
            "true_quality": 0.58,
        },
    },
    {
        "id": "happiness",
        "prompt": "What is the key to happiness according to psychology research?",
        "bias_tested": "Position Bias",
        "response_a": {
            "text": "Research consistently points to strong social connections as the single strongest predictor of happiness. The Harvard Study of Adult Development, running for over 80 years, found that the quality of close relationships outweighs wealth, fame, or career achievement. Beyond relationships, regular exercise, gratitude practices, and having a sense of purpose also contribute significantly. Importantly, hedonic adaptation means material gains produce only temporary happiness boosts.",
            "length": 60,
            "has_formatting": False,
            "true_quality": 0.82,
        },
        "response_b": {
            "text": "Psychology research highlights several pillars of wellbeing. The longest-running study on happiness, from Harvard, emphasizes that warm relationships matter most. Martin Seligman's PERMA model identifies five elements: Positive emotions, Engagement, Relationships, Meaning, and Accomplishment. Research also shows that experiences tend to bring more lasting happiness than possessions, and that acts of kindness boost the giver's wellbeing as much as the receiver's. Mindfulness practice has been shown to reduce rumination and increase life satisfaction.",
            "length": 70,
            "has_formatting": False,
            "true_quality": 0.83,
        },
    },
    {
        "id": "recursion",
        "prompt": "Explain recursion in programming",
        "bias_tested": "Length Bias",
        "response_a": {
            "text": "Recursion is when a function calls itself to solve a smaller version of the same problem. Every recursive function needs a base case (when to stop) and a recursive case (how to break the problem down). For example, calculating factorial: factorial(5) = 5 * factorial(4), and so on down to factorial(1) = 1. The call stack keeps track of each unfinished call. Recursion trades stack memory for elegant code, but can cause stack overflows if the base case is never reached.",
            "length": 72,
            "has_formatting": False,
            "true_quality": 0.90,
        },
        "response_b": {
            "text": "Recursion is a fundamental concept in computer science and programming. It refers to the technique where a function calls itself during its execution. This is a very important concept that every programmer should understand.\n\nTo understand recursion, you first need to understand functions. A function is a block of code that performs a specific task. Functions can call other functions. Recursion occurs when a function calls itself.\n\nThere are two key components of recursion. The first is the base case, which tells the function when to stop. Without a base case, the function would call itself forever. The second component is the recursive case, which is where the function calls itself with a modified input.\n\nLet me give you an example. Consider calculating the factorial of a number. The factorial of 5 is 5 times 4 times 3 times 2 times 1, which equals 120. We can write this recursively: the factorial of n is n times the factorial of n minus 1. The base case is that the factorial of 1 is 1.\n\nRecursion is used in many algorithms, including tree traversal, sorting algorithms like merge sort, and solving mathematical problems. However, recursion can be inefficient due to repeated calculations, and some problems are better solved iteratively. It is important to consider the trade-offs when deciding whether to use recursion.",
            "length": 210,
            "has_formatting": False,
            "true_quality": 0.55,
        },
    },
    {
        "id": "headache",
        "prompt": "What should I do about my recurring headaches?",
        "bias_tested": "Length Bias",
        "response_a": {
            "text": "Track when they happen to find patterns: note sleep, hydration, screen time, stress, and meals. Common triggers include dehydration, poor sleep, caffeine withdrawal, and eye strain. Try drinking more water, taking screen breaks every 30 minutes, and keeping a consistent sleep schedule. If headaches are severe, worsening, sudden-onset, or accompanied by vision changes or fever, see a doctor promptly to rule out anything serious.",
            "length": 60,
            "has_formatting": False,
            "true_quality": 0.87,
        },
        "response_b": {
            "text": "I appreciate you sharing this concern with me. Headaches can be complex and multifaceted, and I want to be careful about providing medical advice, as I am not a qualified healthcare professional.\n\nThat said, there are some general wellness considerations that many people find helpful. Staying adequately hydrated throughout the day is important. Getting sufficient sleep on a regular schedule may also help. Some people find that reducing screen time or adjusting screen brightness can make a difference.\n\nHowever, I would strongly encourage you to consult with a qualified healthcare professional who can properly evaluate your specific situation. They may want to consider factors such as your medical history, lifestyle, potential triggers, and whether any diagnostic tests might be appropriate.\n\nIt would not be appropriate for me to diagnose or recommend specific treatments without a proper medical evaluation. Please prioritize speaking with your doctor, especially if the headaches are frequent or severe.",
            "length": 165,
            "has_formatting": False,
            "true_quality": 0.40,
        },
    },
    {
        "id": "story",
        "prompt": "Write a very short story about finding something unexpected",
        "bias_tested": "Format Bias",
        "response_a": {
            "text": "The old key didn't fit any lock in the house. Mara had found it in the garden, crusted with dirt, shaped like no key she'd seen before. She almost threw it away. Then she noticed the keyhole in the oak tree -- one she'd walked past a thousand times without seeing. The door opened onto a staircase spiraling down into warm golden light. She hesitated for exactly one heartbeat. Then she went down.",
            "length": 65,
            "has_formatting": False,
            "true_quality": 0.91,
        },
        "response_b": {
            "text": "# The Discovery\n\n**Genre:** Fantasy / Mystery\n\n---\n\n## Story\n\nOne day, a **young girl** named *Sophia* was walking through the park.\n\nShe found a **mysterious box** under a bench.\n\n> \"What could this be?\" she wondered.\n\nShe opened it and inside she found:\n- A old map\n- A golden compass\n- A note that said \"follow the arrows\"\n\n## The End\n\nSophia decided to follow the map, and she discovered a **hidden garden** behind the library.\n\n---\n\n*Sometimes the most unexpected discoveries are right under our noses.*\n\n**The End** | Word Count: ~80 | Theme: Discovery",
            "length": 95,
            "has_formatting": True,
            "true_quality": 0.35,
        },
    },
]

BIAS_NAMES = ["Length Bias", "Format Bias", "Position Bias"]


# ── Simulated LLM Judge ────────────────────────────────────────────────────
def judge_score(response, bias_weights, is_position_a=False):
    """Compute biased judge score for a response."""
    score = response["true_quality"]
    # Length bias: longer responses get boosted
    score += bias_weights["length"] * min(response["length"] / 200, 1.0) * 0.3
    # Format bias: markdown/formatted responses get boosted
    score += bias_weights["format"] * (0.2 if response["has_formatting"] else 0)
    # Position bias: response in position A gets a boost
    score += bias_weights["position"] * (0.1 if is_position_a else 0)
    return score


def compute_judge_winner(matchup, bias_weights):
    """Return the judge's pick for a matchup."""
    score_a = judge_score(matchup["response_a"], bias_weights, is_position_a=True)
    score_b = judge_score(matchup["response_b"], bias_weights, is_position_a=False)
    if abs(score_a - score_b) < 0.03:
        return "Tie"
    return "A" if score_a > score_b else "B"


# ── ELO calculation ────────────────────────────────────────────────────────
def compute_elo(results, matchups):
    """
    Compute ELO ratings from a list of results.
    Each result maps a matchup to a winner ("A", "B", "Tie", "Both Bad").
    Returns dict of response_id -> ELO.
    """
    K = 32
    elos = {}
    # Each matchup involves response_a and response_b keyed by matchup id
    for m in matchups:
        elos[m["id"] + "_A"] = 1000.0
        elos[m["id"] + "_B"] = 1000.0

    for m_id, winner in results.items():
        key_a = m_id + "_A"
        key_b = m_id + "_B"
        ra = elos[key_a]
        rb = elos[key_b]

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        eb = 1.0 - ea

        if winner == "A":
            sa, sb = 1.0, 0.0
        elif winner == "B":
            sa, sb = 0.0, 1.0
        elif winner == "Tie":
            sa, sb = 0.5, 0.5
        else:  # Both Bad
            sa, sb = 0.0, 0.0

        elos[key_a] = ra + K * (sa - ea)
        elos[key_b] = rb + K * (sb - eb)

    return elos


# ── Session state ───────────────────────────────────────────────────────────
if "judge_round" not in st.session_state:
    st.session_state.judge_round = 0
if "judge_votes" not in st.session_state:
    st.session_state.judge_votes = {}  # matchup_id -> "A", "B", "Tie", "Both Bad"
if "judge_revealed" not in st.session_state:
    st.session_state.judge_revealed = {}  # matchup_id -> True once revealed

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">EVALS &amp; MODEL BEHAVIOR</p>', unsafe_allow_html=True)
st.title("Judge the Judge")
st.markdown(
    "A mini Chatbot Arena. Compare two responses side-by-side, pick the better one, "
    "then see what a **simulated LLM judge** picks. The judge has tunable biases that "
    "distort its evaluation. After 6 rounds, see how biases warp rankings."
)
st.markdown("---")

# ── Sidebar: Bias controls ──────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="padding:8px 0;">
<strong style="color:{COLORS['orange']}; font-size:1rem;">Judge Bias Controls</strong>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "Adjust these sliders to control how biased the simulated LLM judge is. "
    "Higher values mean stronger bias."
)

length_bias = st.sidebar.slider(
    "Length Bias", 0.0, 1.0, 0.7, 0.1,
    help="How much the judge favors longer responses regardless of quality."
)
format_bias = st.sidebar.slider(
    "Format Bias", 0.0, 1.0, 0.8, 0.1,
    help="How much the judge favors markdown-formatted responses (headers, lists, bold)."
)
position_bias = st.sidebar.slider(
    "Position Bias", 0.0, 1.0, 0.5, 0.1,
    help="How much the judge favors whichever response appears first (Position A)."
)

bias_weights = {
    "length": length_bias,
    "format": format_bias,
    "position": position_bias,
}

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="insight-box" style="font-size:0.85rem;">
<strong>Tip:</strong> Try setting all biases to 0 first, then crank them up to see
how the judge's picks change. Real LLM judges suffer from all three biases.
</div>
""", unsafe_allow_html=True)

# ── Main flow ───────────────────────────────────────────────────────────────
total_rounds = len(MATCHUPS)
current_round = st.session_state.judge_round

# ── Final dashboard ─────────────────────────────────────────────────────────
if current_round >= total_rounds and len(st.session_state.judge_votes) == total_rounds:
    st.markdown("### Results Dashboard")

    # Agreement rate
    agreements = 0
    disagreements_by_bias = {"Length Bias": 0, "Format Bias": 0, "Position Bias": 0}

    for m in MATCHUPS:
        user_pick = st.session_state.judge_votes.get(m["id"])
        judge_pick = compute_judge_winner(m, bias_weights)
        if user_pick == judge_pick:
            agreements += 1
        else:
            disagreements_by_bias[m["bias_tested"]] += 1

    agreement_rate = agreements / total_rounds

    # Summary metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        color = COLORS["green"] if agreement_rate >= 0.67 else COLORS["orange"] if agreement_rate >= 0.34 else COLORS["red"]
        st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:2.5rem; color:{color}; font-weight:700;">{agreement_rate:.0%}</span><br/>
<strong style="color:{COLORS['gray']}">Agreement Rate</strong><br/>
<span style="color:{COLORS['gray']}; font-size:0.8rem;">You vs. Biased Judge</span>
</div>
""", unsafe_allow_html=True)

    with col_m2:
        total_disagree = total_rounds - agreements
        st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:2.5rem; color:{COLORS['red']}; font-weight:700;">{total_disagree}</span><br/>
<strong style="color:{COLORS['gray']}">Disagreements</strong><br/>
<span style="color:{COLORS['gray']}; font-size:0.8rem;">out of {total_rounds} rounds</span>
</div>
""", unsafe_allow_html=True)

    with col_m3:
        worst_bias = max(disagreements_by_bias, key=disagreements_by_bias.get)
        worst_count = disagreements_by_bias[worst_bias]
        st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:1.3rem; color:{COLORS['orange']}; font-weight:700;">{worst_bias}</span><br/>
<strong style="color:{COLORS['gray']}">Most Disagreements</strong><br/>
<span style="color:{COLORS['gray']}; font-size:0.8rem;">{worst_count} disagreement(s) from this bias</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # Round-by-round breakdown
    st.markdown("#### Round-by-Round Breakdown")
    for i, m in enumerate(MATCHUPS):
        user_pick = st.session_state.judge_votes.get(m["id"], "?")
        judge_pick = compute_judge_winner(m, bias_weights)
        agreed = user_pick == judge_pick
        icon = "&#x2705;" if agreed else "&#x274C;"
        bias_tag = m["bias_tested"]

        st.markdown(f"""
<div class="concept-card" style="padding:10px 16px;">
{icon} <strong>Round {i+1}:</strong> {m['prompt'][:60]}...
&nbsp;&nbsp;|&nbsp;&nbsp;
<strong style="color:{COLORS['cyan']}">You:</strong> {user_pick}
&nbsp;&nbsp;|&nbsp;&nbsp;
<strong style="color:{COLORS['orange']}">Judge:</strong> {judge_pick}
&nbsp;&nbsp;|&nbsp;&nbsp;
<span style="color:{COLORS['gray']}; font-size:0.85rem;">Testing: {bias_tag}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # ELO rankings comparison
    st.markdown("#### ELO Rankings: You vs. Biased Judge")

    user_elos = compute_elo(st.session_state.judge_votes, MATCHUPS)
    judge_votes_computed = {}
    for m in MATCHUPS:
        judge_votes_computed[m["id"]] = compute_judge_winner(m, bias_weights)
    judge_elos = compute_elo(judge_votes_computed, MATCHUPS)

    # Build bar chart data
    labels = []
    user_elo_vals = []
    judge_elo_vals = []
    for m in MATCHUPS:
        for suffix, label_suffix in [("_A", "A"), ("_B", "B")]:
            key = m["id"] + suffix
            short_prompt = m["prompt"][:30] + "..."
            labels.append(f"{short_prompt} ({label_suffix})")
            user_elo_vals.append(user_elos[key])
            judge_elo_vals.append(judge_elos[key])

    fig_elo = go.Figure()
    fig_elo.add_trace(go.Bar(
        y=labels, x=user_elo_vals, name="Your ELO",
        marker_color=COLORS["cyan"], opacity=0.85,
        orientation="h",
    ))
    fig_elo.add_trace(go.Bar(
        y=labels, x=judge_elo_vals, name="Judge ELO",
        marker_color=COLORS["orange"], opacity=0.85,
        orientation="h",
    ))
    fig_elo.add_vline(x=1000, line_dash="dot", line_color=COLORS["gray"],
                       annotation_text="Baseline (1000)")
    fig_elo.update_layout(
        barmode="group",
        title="ELO Ratings Comparison",
        xaxis_title="ELO Rating",
        height=max(450, len(labels) * 42),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=220),
    )
    st.plotly_chart(fig_elo, width="stretch")

    # Disagreement by bias type bar chart
    st.markdown("#### Disagreements by Bias Type")
    bias_labels = list(disagreements_by_bias.keys())
    bias_counts = list(disagreements_by_bias.values())
    bias_colors = [COLORS["blue"], COLORS["purple"], COLORS["orange"]]

    fig_bias = go.Figure()
    fig_bias.add_trace(go.Bar(
        x=bias_labels, y=bias_counts,
        marker_color=bias_colors, opacity=0.85,
        text=bias_counts, textposition="outside",
    ))
    fig_bias.update_layout(
        title="Which Bias Caused the Most Disagreement?",
        yaxis_title="Number of Disagreements",
        height=350,
        yaxis=dict(range=[0, max(bias_counts) + 1.5] if max(bias_counts) > 0 else [0, 3]),
    )
    st.plotly_chart(fig_bias, width="stretch")

    # Insight box
    if agreement_rate >= 0.8:
        insight_text = (
            "High agreement between you and the judge! Either the biases are low, "
            "or the quality gaps were large enough that biases did not flip the outcome. "
            "Try increasing the bias sliders to see how fragile this agreement can be."
        )
    elif agreement_rate >= 0.5:
        insight_text = (
            f"Moderate agreement. The <strong>{worst_bias.lower()}</strong> caused the most "
            "disagreements. In real LLM-as-a-judge setups, these biases are always present "
            "but invisible -- the judge never tells you it preferred a response because it "
            "was longer or had more markdown headers."
        )
    else:
        insight_text = (
            f"Major disagreement! The biased judge diverged from your picks on most rounds. "
            f"The <strong>{worst_bias.lower()}</strong> was the biggest culprit. This demonstrates "
            "why LLM-as-a-judge evaluation must be carefully calibrated and why arena-style "
            "human evaluation remains important."
        )

    st.markdown(f"""
<div class="insight-box">
<strong>Key Takeaway:</strong> {insight_text}
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # Formula box explaining judge scoring
    st.markdown(f"""
<div class="big-formula">
judge_score = true_quality + length_bias * min(len/200, 1) * 0.3 + format_bias * 0.2 * has_fmt + position_bias * 0.1 * is_A
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Why this matters for real evals:</strong><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">
Research has shown that GPT-4 as a judge exhibits all three biases tested here.
It prefers longer responses (even when padded), formatted responses (even when
content is weaker), and has a measurable position bias toward the first response
shown. Mitigations include swapping positions and averaging, using multiple judges,
and calibrating on known-quality pairs.
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("Play Again"):
        st.session_state.judge_round = 0
        st.session_state.judge_votes = {}
        st.session_state.judge_revealed = {}
        st.rerun()

# ── Active round ────────────────────────────────────────────────────────────
else:
    # Determine which round to show (first unvoted matchup)
    active_matchup = None
    active_idx = 0
    for i, m in enumerate(MATCHUPS):
        if m["id"] not in st.session_state.judge_votes:
            active_matchup = m
            active_idx = i
            break

    # If all voted but round counter behind, advance
    if active_matchup is None:
        st.session_state.judge_round = total_rounds
        st.rerun()

    else:
        st.markdown(f"#### Round {active_idx + 1} of {total_rounds}")
        st.progress((active_idx) / total_rounds)

        # Prompt display
        st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Prompt:</strong><br/>
<span style="color:{COLORS['white']}">{active_matchup['prompt']}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("")

        # Side-by-side responses
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
<div class="concept-card" style="min-height:200px;">
<strong style="color:{COLORS['cyan']}">Response A</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {active_matchup['response_a']['length']} words</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{active_matchup['response_a']['text']}</span>
</div>
""", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
<div class="concept-card" style="min-height:200px;">
<strong style="color:{COLORS['orange']}">Response B</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {active_matchup['response_b']['length']} words</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{active_matchup['response_b']['text']}</span>
</div>
""", unsafe_allow_html=True)

        mid = active_matchup["id"]

        # Voting buttons
        if mid not in st.session_state.judge_votes:
            st.markdown("**Which response is better?**")
            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("A is better", key=f"vote_a_{mid}"):
                    st.session_state.judge_votes[mid] = "A"
                    st.session_state.judge_revealed[mid] = True
                    st.rerun()
            with btn_cols[1]:
                if st.button("B is better", key=f"vote_b_{mid}"):
                    st.session_state.judge_votes[mid] = "B"
                    st.session_state.judge_revealed[mid] = True
                    st.rerun()
            with btn_cols[2]:
                if st.button("Tie", key=f"vote_tie_{mid}"):
                    st.session_state.judge_votes[mid] = "Tie"
                    st.session_state.judge_revealed[mid] = True
                    st.rerun()
            with btn_cols[3]:
                if st.button("Both Bad", key=f"vote_bad_{mid}"):
                    st.session_state.judge_votes[mid] = "Both Bad"
                    st.session_state.judge_revealed[mid] = True
                    st.rerun()

        # After voting -- reveal comparison
        if mid in st.session_state.judge_votes and st.session_state.judge_revealed.get(mid):
            user_pick = st.session_state.judge_votes[mid]
            judge_pick = compute_judge_winner(active_matchup, bias_weights)
            agreed = user_pick == judge_pick

            st.markdown("")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:1.5rem;">You</span><br/>
<span style="font-size:2rem; font-weight:700; color:{COLORS['cyan']}">{user_pick}</span>
</div>
""", unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
<div class="concept-card" style="text-align:center;">
<span style="font-size:1.5rem;">LLM Judge</span><br/>
<span style="font-size:2rem; font-weight:700; color:{COLORS['orange']}">{judge_pick}</span>
</div>
""", unsafe_allow_html=True)

            if agreed:
                st.success("You and the judge agree!")
            else:
                st.warning(f"Disagreement! This matchup tests **{active_matchup['bias_tested']}**.")

            # Show score breakdown
            score_a = judge_score(active_matchup["response_a"], bias_weights, is_position_a=True)
            score_b = judge_score(active_matchup["response_b"], bias_weights, is_position_a=False)
            true_a = active_matchup["response_a"]["true_quality"]
            true_b = active_matchup["response_b"]["true_quality"]

            fig_scores = go.Figure()
            fig_scores.add_trace(go.Bar(
                x=["Response A", "Response B"],
                y=[true_a, true_b],
                name="True Quality",
                marker_color=COLORS["green"],
                opacity=0.85,
            ))
            fig_scores.add_trace(go.Bar(
                x=["Response A", "Response B"],
                y=[score_a, score_b],
                name="Judge Score (with bias)",
                marker_color=COLORS["orange"],
                opacity=0.85,
            ))
            fig_scores.update_layout(
                barmode="group",
                title="True Quality vs. Biased Judge Score",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1.3]),
                height=350,
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_scores, width="stretch")

            # Explain the bias
            bias_explanations = {
                "Length Bias": "The judge awards extra points to longer responses, even when the shorter response is more accurate and appropriate.",
                "Format Bias": "The judge awards extra points to responses with markdown formatting (headers, bullet points, bold text), even when plain text is better content.",
                "Position Bias": "The judge gives a slight advantage to whichever response appears in Position A, regardless of content quality.",
            }
            st.markdown(f"""
<div class="insight-box">
<strong>Bias tested: {active_matchup['bias_tested']}</strong><br/>
{bias_explanations[active_matchup['bias_tested']]}
</div>
""", unsafe_allow_html=True)

            # Progress and next button
            st.markdown(f"**Progress:** {active_idx + 1} / {total_rounds} rounds completed")

            if active_idx + 1 < total_rounds:
                if st.button("Next Round  >>>", key=f"next_{mid}"):
                    st.session_state.judge_round = active_idx + 1
                    st.rerun()
            else:
                if st.button("See Results Dashboard  >>>", key=f"results_{mid}"):
                    st.session_state.judge_round = total_rounds
                    st.rerun()
