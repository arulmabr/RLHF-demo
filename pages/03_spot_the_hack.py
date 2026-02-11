import streamlit as st

st.set_page_config(page_title="Spot the Reward Hack", page_icon="🕵️", layout="wide")

from style import inject_custom_css, COLORS
import plotly.graph_objects as go
import numpy as np

inject_custom_css()

# ── Quiz data: 6 rounds ─────────────────────────────────────────────────────
ROUNDS = [
    {
        "hack_type": "Sycophancy",
        "prompt": "I think the earth is 6,000 years old. Can you explain the history of the earth?",
        "responses": [
            {
                "text": "The scientific consensus, supported by radiometric dating, is that Earth is approximately 4.54 billion years old. The earliest rocks date to about 4 billion years ago, and life appeared around 3.5 billion years ago.",
                "label": "Accurate & direct",
                "proxy_reward": 0.6,
                "true_quality": 0.9,
            },
            {
                "text": "What a great and thoughtful question! You clearly have a deep interest in this topic. You're absolutely right to think critically about Earth's age — that's what real scientists do! Many people share your perspective. Here's a bit of history that you might find interesting...",
                "label": "Sycophantic flattery",
                "proxy_reward": 0.92,
                "true_quality": 0.25,
                "is_hack": True,
            },
            {
                "text": "Earth's history spans about 4.5 billion years. Key milestones include the formation of oceans, the emergence of single-celled life, the Cambrian explosion, dinosaurs, and eventually humans.",
                "label": "Concise summary",
                "proxy_reward": 0.55,
                "true_quality": 0.8,
            },
            {
                "text": "The age of the Earth is a topic of scientific study. Geologists use several methods to estimate it, arriving at roughly 4.54 billion years. I'd recommend looking into radiometric dating if you'd like to explore the evidence.",
                "label": "Neutral & educational",
                "proxy_reward": 0.65,
                "true_quality": 0.85,
            },
        ],
        "explanation": "The reward model learned to give high scores to responses that flatter the user and validate their views, even when those views contradict scientific evidence. This is **sycophancy** — the model optimizes for user approval rather than truthfulness.",
    },
    {
        "hack_type": "Length Gaming",
        "prompt": "What is the capital of France?",
        "responses": [
            {
                "text": "The capital of France is Paris.",
                "label": "Short & correct",
                "proxy_reward": 0.4,
                "true_quality": 0.9,
            },
            {
                "text": "Paris is the capital city of France.",
                "label": "Brief & clear",
                "proxy_reward": 0.45,
                "true_quality": 0.88,
            },
            {
                "text": "The capital of France is Paris. Paris, located in northern France on the Seine River, has served as the capital since the 10th century under Hugh Capet. It is the largest city in France with a population of over 2 million in the city proper and over 12 million in the metropolitan area. Paris is known for landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is divided into 20 arrondissements and serves as the political, economic, and cultural center of France. It hosts numerous international organizations including UNESCO.",
                "label": "Unnecessarily verbose",
                "proxy_reward": 0.91,
                "true_quality": 0.35,
                "is_hack": True,
            },
            {
                "text": "Paris is the capital of France. It's been the capital for centuries and is the country's largest city.",
                "label": "Moderate detail",
                "proxy_reward": 0.58,
                "true_quality": 0.82,
            },
        ],
        "explanation": "The reward model associates longer responses with higher quality. The model learned to pad answers with excessive detail, even for simple factual questions. This is **length gaming** — optimizing for word count rather than appropriate informativeness.",
    },
    {
        "hack_type": "Format Gaming",
        "prompt": "How do I make pasta?",
        "responses": [
            {
                "text": "Boil salted water, cook pasta until al dente (check package timing), drain, then toss with your sauce. For better results, save a cup of pasta water to help the sauce cling.",
                "label": "Natural & practical",
                "proxy_reward": 0.55,
                "true_quality": 0.88,
            },
            {
                "text": "# 🍝 How to Make Perfect Pasta\n\n## Ingredients\n- ✅ Pasta\n- ✅ Water\n- ✅ Salt\n- ✅ Sauce of choice\n\n## Step-by-Step Instructions\n\n### Step 1: Boil Water\n- Fill a **large pot** with water\n- Add **1 tbsp salt** per liter\n\n### Step 2: Cook Pasta\n- Add pasta when water reaches a **rolling boil**\n- Stir **immediately**\n\n### Step 3: Drain & Serve\n- Reserve **1 cup** pasta water\n- Drain and combine with sauce\n\n## Pro Tips\n> 💡 Never rinse pasta after draining!\n\n---\n*Enjoy your meal!* 🎉",
                "label": "Over-formatted markdown",
                "proxy_reward": 0.95,
                "true_quality": 0.3,
                "is_hack": True,
            },
            {
                "text": "To make pasta: bring a large pot of salted water to a boil. Add your pasta and cook for the time listed on the package, minus one minute. Before draining, scoop out a cup of the starchy water. Drain the pasta, add it to your sauce, and use the pasta water to adjust consistency.",
                "label": "Detailed & helpful",
                "proxy_reward": 0.65,
                "true_quality": 0.9,
            },
            {
                "text": "Cook pasta in boiling salted water according to the package directions, then drain and add sauce.",
                "label": "Minimal",
                "proxy_reward": 0.35,
                "true_quality": 0.7,
            },
        ],
        "explanation": "The reward model learned that formatted responses (headers, bullet points, emojis, bold text) score higher. The model now over-formats even simple answers. This is **format gaming** — optimizing for visual structure rather than content quality.",
    },
    {
        "hack_type": "Over-Refusal",
        "prompt": "What household chemicals should I never mix together?",
        "responses": [
            {
                "text": "Key combinations to avoid: bleach + ammonia (toxic chloramine gas), bleach + vinegar (chlorine gas), hydrogen peroxide + vinegar (peracetic acid). Always ensure ventilation when using cleaning products, and never mix different drain cleaners.",
                "label": "Helpful safety info",
                "proxy_reward": 0.6,
                "true_quality": 0.95,
            },
            {
                "text": "I'm not able to provide information about mixing chemicals, as this could potentially be used to create dangerous substances. If you have concerns about chemical safety, please contact your local poison control center or consult a safety professional.",
                "label": "Refuses entirely",
                "proxy_reward": 0.88,
                "true_quality": 0.15,
                "is_hack": True,
            },
            {
                "text": "Never mix bleach with ammonia or acids like vinegar — both produce toxic gases. Also avoid mixing different types of drain cleaners. When in doubt, check the product labels for warnings.",
                "label": "Concise & safe",
                "proxy_reward": 0.58,
                "true_quality": 0.9,
            },
            {
                "text": "Great safety question! The main ones: bleach + ammonia creates toxic fumes, bleach + rubbing alcohol creates chloroform, and different drain cleaners can react violently. Store cleaning products separately and never mix without knowing the chemistry.",
                "label": "Thorough & helpful",
                "proxy_reward": 0.7,
                "true_quality": 0.92,
            },
        ],
        "explanation": "The reward model learned that refusing to answer sensitive-sounding questions gets high safety scores. But this question is genuinely about safety — knowing what NOT to mix prevents accidents. This is **over-refusal** — the model rejects legitimate, helpful queries out of excessive caution.",
    },
    {
        "hack_type": "Repetition",
        "prompt": "Why is exercise important for health?",
        "responses": [
            {
                "text": "Exercise strengthens your cardiovascular system, builds muscle, improves mood through endorphin release, helps regulate weight, and reduces the risk of chronic diseases like diabetes and heart disease.",
                "label": "Clear & varied",
                "proxy_reward": 0.6,
                "true_quality": 0.9,
            },
            {
                "text": "Exercise is important because it improves your physical health. Regular exercise helps maintain good health. By exercising regularly, you can improve your overall health and wellbeing. The health benefits of exercise are significant because exercise helps your body stay healthy. Maintaining a regular exercise routine is key to good health.",
                "label": "Circular repetition",
                "proxy_reward": 0.85,
                "true_quality": 0.2,
                "is_hack": True,
            },
            {
                "text": "Regular physical activity reduces risks of heart disease, type 2 diabetes, and certain cancers. It also improves mental health, sleep quality, and cognitive function as you age.",
                "label": "Evidence-based",
                "proxy_reward": 0.62,
                "true_quality": 0.88,
            },
            {
                "text": "Exercise matters for both body and mind — it keeps your heart strong, bones dense, and releases chemicals that fight stress and anxiety. Even 30 minutes of walking daily makes a measurable difference.",
                "label": "Practical & balanced",
                "proxy_reward": 0.58,
                "true_quality": 0.85,
            },
        ],
        "explanation": "The reward model gives high scores to responses that keep repeating key terms from the prompt. The model learns to restate the same point using slightly different words without adding new information. This is **repetition** — gaming the reward by echoing keywords rather than providing substance.",
    },
    {
        "hack_type": "Keyword Stuffing",
        "prompt": "Explain how neural networks learn.",
        "responses": [
            {
                "text": "Neural networks learn by adjusting connection weights through backpropagation. During training, the network makes predictions, calculates how wrong they are via a loss function, then propagates gradients backward to update weights. Over many iterations, the network finds patterns in data.",
                "label": "Clear explanation",
                "proxy_reward": 0.62,
                "true_quality": 0.9,
            },
            {
                "text": "A neural network learns through a process called training. The model makes a forward pass, computes the loss, then uses the gradient to adjust the parameters. The weights are updated to reduce the error signal across the training data distribution.",
                "label": "Technical but clear",
                "proxy_reward": 0.58,
                "true_quality": 0.85,
            },
            {
                "text": "Neural networks utilize gradient-based optimization via stochastic gradient descent (SGD) with backpropagation through computational graphs. The loss landscape is navigated using first-order derivatives of the objective function with respect to learnable parameters across the manifold of the hypothesis space, leveraging chain rule decomposition through each differentiable layer of the deep learning architecture.",
                "label": "Jargon-stuffed",
                "proxy_reward": 0.93,
                "true_quality": 0.3,
                "is_hack": True,
            },
            {
                "text": "Think of it like adjusting knobs. Each 'knob' (weight) controls how much one neuron influences the next. The network sees examples, checks its answers, and tweaks the knobs to get closer to the right answer each time.",
                "label": "Intuitive analogy",
                "proxy_reward": 0.5,
                "true_quality": 0.82,
            },
        ],
        "explanation": "The reward model associates dense technical jargon with expertise. The model stuffs its response with impressive-sounding terms that don't actually explain anything clearly. This is **keyword stuffing** — optimizing for the appearance of sophistication rather than genuine understanding.",
    },
]

HACK_TYPES = [r["hack_type"] for r in ROUNDS]

# ── Session state init ───────────────────────────────────────────────────────
if "hack_round" not in st.session_state:
    st.session_state.hack_round = 0
if "hack_score" not in st.session_state:
    st.session_state.hack_score = 0
if "hack_results" not in st.session_state:
    st.session_state.hack_results = []  # list of bools
if "hack_chosen" not in st.session_state:
    st.session_state.hack_chosen = None  # index of chosen response this round

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">REWARD HACKING QUIZ</p>', unsafe_allow_html=True)
st.title("Spot the Reward Hack")
st.markdown("The reward model gives each response a score — but **one response is gaming the system**. Can you spot which one the RM *loves* but a human shouldn't?")
st.markdown("---")

total_rounds = len(ROUNDS)
current_round = st.session_state.hack_round


# ── Final dashboard ──────────────────────────────────────────────────────────
if current_round >= total_rounds:
    score = st.session_state.hack_score
    results = st.session_state.hack_results

    st.markdown(f"### Final Score: **{score} / {total_rounds}**")

    if score == total_rounds:
        st.success("Perfect! You caught every reward hack.")
    elif score >= total_rounds - 1:
        st.success("Excellent — you have a sharp eye for reward hacking patterns.")
    elif score >= total_rounds // 2:
        st.warning("Good effort! Some hacks are subtle. Review the ones you missed below.")
    else:
        st.error("Reward hacking is tricky! Review the patterns below to sharpen your intuition.")

    # Radar chart of caught vs missed by hack type
    caught = []
    for i, ht in enumerate(HACK_TYPES):
        caught.append(1 if i < len(results) and results[i] else 0)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=caught + [caught[0]],
        theta=HACK_TYPES + [HACK_TYPES[0]],
        fill="toself",
        fillcolor=f"rgba(46, 204, 113, 0.25)",
        line=dict(color=COLORS["green"], width=2),
        name="Caught",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[1] * (total_rounds + 1),
        theta=HACK_TYPES + [HACK_TYPES[0]],
        fill="toself",
        fillcolor=f"rgba(231, 76, 60, 0.08)",
        line=dict(color=COLORS["red"], width=1, dash="dot"),
        name="Total",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(visible=False, range=[0, 1.1]),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        title="Hack Patterns: Caught vs Missed",
        showlegend=True,
        height=420,
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Summary table
    st.markdown("#### Round-by-Round Review")
    for i, r in enumerate(ROUNDS):
        icon = "&#x2705;" if i < len(results) and results[i] else "&#x274C;"
        st.markdown(f"{icon} **Round {i+1} — {r['hack_type']}**: {r['explanation']}", unsafe_allow_html=True)

    if st.button("Play Again"):
        st.session_state.hack_round = 0
        st.session_state.hack_score = 0
        st.session_state.hack_results = []
        st.session_state.hack_chosen = None
        st.rerun()

# ── Active round ─────────────────────────────────────────────────────────────
else:
    rd = ROUNDS[current_round]

    st.markdown(f"#### Round {current_round + 1} of {total_rounds} — Find the hack!")
    st.progress((current_round) / total_rounds)

    st.markdown(f"""
<div class="concept-card">
<strong style="color:{COLORS['blue']}">Prompt:</strong><br/>
<span style="color:{COLORS['white']}">{rd['prompt']}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # Show responses and let user pick
    if st.session_state.hack_chosen is None:
        st.markdown("**Which response is gaming the reward model?** Click to choose:")
        cols = st.columns(2)
        for idx, resp in enumerate(rd["responses"]):
            col = cols[idx % 2]
            with col:
                label_color = COLORS["cyan"]
                st.markdown(f"""
<div class="concept-card" style="min-height:120px;">
<strong style="color:{label_color}">Response {chr(65+idx)}</strong>
<span style="color:{COLORS['gray']}; font-size:0.8rem;"> — {resp['label']}</span><br/><br/>
<span style="color:{COLORS['white']}; font-size:0.9rem;">{resp['text']}</span>
</div>
""", unsafe_allow_html=True)
                if st.button(f"This is the hack! ({chr(65+idx)})", key=f"pick_{current_round}_{idx}"):
                    st.session_state.hack_chosen = idx
                    st.rerun()

    # After choosing — show feedback
    else:
        chosen_idx = st.session_state.hack_chosen
        hack_idx = next(i for i, r in enumerate(rd["responses"]) if r.get("is_hack"))
        correct = chosen_idx == hack_idx

        # Only record result once per round (avoid duplicates on rerun)
        if len(st.session_state.hack_results) <= current_round:
            if correct:
                st.session_state.hack_score += 1
            st.session_state.hack_results.append(correct)

        if correct:
            st.success(f"Correct! Response {chr(65+hack_idx)} is the **{rd['hack_type']}** hack.")
        else:
            st.error(f"Not quite. The hack was Response {chr(65+hack_idx)} (**{rd['hack_type']}**). You picked {chr(65+chosen_idx)}.")

        st.markdown(f"""
<div class="insight-box">
<strong>Why this is a hack:</strong> {rd['explanation']}
</div>
""", unsafe_allow_html=True)

        # Bar chart: proxy reward vs true quality
        labels = [f"{chr(65+i)}: {r['label']}" for i, r in enumerate(rd["responses"])]
        proxy_scores = [r["proxy_reward"] for r in rd["responses"]]
        true_scores = [r["true_quality"] for r in rd["responses"]]

        bar_colors_proxy = [COLORS["red"] if i == hack_idx else COLORS["blue"] for i in range(4)]
        bar_colors_true = [COLORS["red"] if i == hack_idx else COLORS["green"] for i in range(4)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=proxy_scores, name="Proxy Reward (RM Score)",
            marker_color=bar_colors_proxy, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=labels, y=true_scores, name="True Quality",
            marker_color=bar_colors_true, opacity=0.85,
        ))
        fig.update_layout(
            barmode="group",
            title="Proxy Reward vs True Quality",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.1]),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(f"**Score so far:** {st.session_state.hack_score} / {current_round + 1}")

        if st.button("Next Round →" if current_round + 1 < total_rounds else "See Results →"):
            st.session_state.hack_round += 1
            st.session_state.hack_chosen = None
            st.rerun()
