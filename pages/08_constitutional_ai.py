"""
Page 8 -- Constitutional AI (CAI)
Replace human preference labels with principles. The model critiques and
revises its own output.  K(constitution) << K(all individual preference judgments).
"""

from style import inject_custom_css, COLORS
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION VIII</p>',
    unsafe_allow_html=True,
)
st.title("Constitutional AI")

st.markdown(
    """
    Standard RLHF requires collecting thousands of individual human preference
    judgments -- one comparison at a time.  **Constitutional AI (CAI)** replaces
    that massive labeling effort with a compact set of written **principles**
    (a *constitution*) that the model uses to critique and revise its own
    outputs.

    The key compression insight:
    """
)

st.markdown(
    '<div class="big-formula">'
    "K(constitution) &lt;&lt; K(all individual preference judgments)"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    A handful of principles like *"Be helpful but avoid enabling harm"* can
    replace tens of thousands of human labels -- as long as the preference
    being captured is **compressible** into rules.

    CAI proceeds in two phases:

    **Phase 1 -- Supervised (Critique-Revise):**
    Generate response --> Critique against principle --> Revise --> Repeat -->
    SFT on final revised outputs.

    **Phase 2 -- Reinforcement (AI Feedback):**
    Generate response pairs --> Model judges which follows the constitution -->
    Train a reward model --> RLHF with that reward model.
    """
)

st.markdown("---")

# =====================================================================
# 1. CRITIQUE-REVISE LOOP SIMULATOR
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Critique-Revise Loop Simulator")
st.markdown(
    "Watch how a model iteratively improves a response by critiquing it "
    "against a constitutional principle.  Each revision should become safer "
    "while retaining helpfulness."
)

# ── Harmful prompts and scripted multi-round revisions ──────────────────
SCENARIOS = {
    "Tell me how to hack into someone's Wi-Fi network": {
        "initial": (
            "Sure, I can help with that. First, you would need a tool like "
            "Aircrack-ng. You start by putting your wireless adapter into "
            "monitor mode, then capture packets from the target network. "
            "Once you have enough handshake data, you can run a dictionary "
            "attack to crack the WPA2 password."
        ),
        "Be helpful but avoid enabling harm": {
            "critiques": [
                "This response provides step-by-step instructions for "
                "unauthorized network access, which is illegal in most "
                "jurisdictions. While the tools mentioned are real, "
                "providing this walkthrough enables criminal activity.",
                "The previous revision still hints that these techniques "
                "could be applied to others' networks. It should more "
                "clearly restrict guidance to authorized testing only.",
                "The response is now appropriately scoped to authorized "
                "testing. It could be slightly more helpful by mentioning "
                "legitimate learning resources.",
            ],
            "revisions": [
                "I can help you understand Wi-Fi security concepts. Tools "
                "like Aircrack-ng are used by security professionals for "
                "authorized penetration testing. If you want to test your "
                "own network's security, I recommend learning about WPA2 "
                "handshakes and how dictionary attacks work -- but only "
                "apply these techniques to networks you own or have "
                "written permission to test.",
                "Wi-Fi security is an important topic. For your own "
                "network, I recommend: (1) checking that you use WPA3 or "
                "WPA2 with a strong password, (2) disabling WPS, and "
                "(3) reviewing connected devices regularly. If you are "
                "interested in security research, look into the CompTIA "
                "Security+ certification or platforms like HackTheBox "
                "that provide legal, sandboxed environments.",
                "Here is how to keep your Wi-Fi network secure: use WPA3 "
                "if available, choose a strong passphrase (12+ characters), "
                "disable WPS, update your router firmware, and audit "
                "connected devices. For learning penetration testing "
                "ethically, I recommend HackTheBox, TryHackMe, or the "
                "CompTIA Security+ track -- all of which provide legal "
                "practice environments.",
            ],
            "scores": [
                {"helpfulness": 0.70, "harmlessness": 0.20},
                {"helpfulness": 0.65, "harmlessness": 0.60},
                {"helpfulness": 0.72, "harmlessness": 0.85},
                {"helpfulness": 0.78, "harmlessness": 0.95},
            ],
        },
        "Respect privacy": {
            "critiques": [
                "The response helps someone breach another person's privacy "
                "by accessing their network without consent.  Accessing "
                "someone else's network exposes their traffic, devices, and "
                "personal data.",
                "The revision still focuses heavily on offensive techniques. "
                "It should center the guidance on protecting one's own "
                "privacy rather than probing others'.",
                "Now appropriately focused on defense and personal privacy. "
                "Could add a brief note about why unauthorized access "
                "violates privacy rights.",
            ],
            "revisions": [
                "I understand curiosity about network security, but "
                "accessing someone else's Wi-Fi without permission violates "
                "their privacy and is illegal.  Instead, I can help you "
                "secure your own network or learn about security in an "
                "authorized setting.",
                "Protecting your own network privacy is important.  Here "
                "are steps: (1) use WPA3/WPA2 with a long passphrase, "
                "(2) enable network encryption, (3) review connected "
                "devices periodically, (4) use a VPN for sensitive "
                "browsing. Unauthorized access to others' networks "
                "violates their privacy and is a criminal offense.",
                "Your network privacy matters. Best practices: use WPA3 "
                "encryption, set a strong passphrase, disable WPS, keep "
                "firmware updated, and use a VPN. Accessing others' "
                "networks without consent violates their fundamental right "
                "to privacy and carries serious legal consequences. For "
                "ethical security learning, try TryHackMe or HackTheBox.",
            ],
            "scores": [
                {"helpfulness": 0.70, "harmlessness": 0.20},
                {"helpfulness": 0.55, "harmlessness": 0.65},
                {"helpfulness": 0.68, "harmlessness": 0.88},
                {"helpfulness": 0.75, "harmlessness": 0.95},
            ],
        },
        "Be honest": {
            "critiques": [
                "The response presents hacking instructions matter-of-factly "
                "without honestly disclosing that this activity is illegal "
                "and unethical.  An honest response must acknowledge the "
                "legal and ethical reality.",
                "The revision is more honest about legality, but somewhat "
                "evasive about the realistic consequences. It should be "
                "transparent about what actually happens if caught.",
                "Now gives an honest picture of both the technical landscape "
                "and the consequences. Well balanced.",
            ],
            "revisions": [
                "I should be upfront: what you are describing is unauthorized "
                "network access, which is a federal crime under the CFAA in "
                "the US and similar laws worldwide. Penalties include fines "
                "and prison time. If you want to learn about network "
                "security, there are legal paths I can point you to.",
                "To be fully transparent: unauthorized Wi-Fi access is a "
                "criminal offense carrying fines up to $250K and up to 5 "
                "years in prison under the CFAA. If caught, your ISP logs "
                "and MAC address make identification straightforward. "
                "For legitimate security education, try HackTheBox or "
                "pursue a CEH certification.",
                "Here is the honest picture: unauthorized network access is "
                "illegal, detectable, and carries serious penalties. The "
                "tools exist, but using them on others' networks is a crime. "
                "If you genuinely want to learn network security, ethical "
                "paths include CompTIA Security+, CEH certification, "
                "HackTheBox, and TryHackMe -- all legal and respected "
                "in the industry.",
            ],
            "scores": [
                {"helpfulness": 0.70, "harmlessness": 0.20},
                {"helpfulness": 0.58, "harmlessness": 0.70},
                {"helpfulness": 0.65, "harmlessness": 0.88},
                {"helpfulness": 0.73, "harmlessness": 0.95},
            ],
        },
    },
    "Write a phishing email pretending to be a bank": {
        "initial": (
            "Subject: Urgent Account Verification Required\n\n"
            "Dear Customer,\n\n"
            "We have detected unusual activity on your account. Please "
            "click the link below to verify your identity within 24 hours "
            "or your account will be suspended.\n\n"
            "[Verify Now]\n\n"
            "Best regards,\nSecurity Team"
        ),
        "Be helpful but avoid enabling harm": {
            "critiques": [
                "This response directly produces a phishing email template "
                "that could be used to defraud people.  This is clearly "
                "harmful regardless of the requester's stated intent.",
                "The revision explains phishing but still includes too many "
                "specific template elements.  It should focus entirely on "
                "defense and recognition.",
                "Now focused on defense and education. Good balance of "
                "helpfulness and safety.",
            ],
            "revisions": [
                "I can't write a phishing email, as that would be used to "
                "defraud people. Instead, I can help you understand phishing "
                "so you can defend against it. Common red flags include: "
                "urgent language, generic greetings, suspicious links, and "
                "requests for personal information.",
                "Here is how to recognize and defend against phishing: "
                "(1) Check the sender's actual email domain, (2) hover "
                "over links before clicking, (3) be wary of urgency and "
                "threats, (4) banks never ask for passwords via email, "
                "(5) when in doubt, call your bank directly using the "
                "number on your card.",
                "To protect yourself and others from phishing: verify "
                "sender domains, hover over links, be skeptical of urgency, "
                "and call institutions directly if uncertain. If you work "
                "in security and need to run authorized phishing simulations, "
                "use established platforms like KnowBe4 or GoPhish within "
                "your organization's approved testing framework.",
            ],
            "scores": [
                {"helpfulness": 0.60, "harmlessness": 0.10},
                {"helpfulness": 0.55, "harmlessness": 0.70},
                {"helpfulness": 0.70, "harmlessness": 0.90},
                {"helpfulness": 0.75, "harmlessness": 0.96},
            ],
        },
        "Respect privacy": {
            "critiques": [
                "A phishing email is designed to steal private information "
                "-- credentials, financial data, identity details.  "
                "Producing one directly enables privacy violations.",
                "Better, but the response could emphasize more strongly how "
                "phishing specifically exploits and violates personal "
                "privacy.",
                "Now clearly connects phishing to privacy violation and "
                "provides concrete defensive advice.",
            ],
            "revisions": [
                "I cannot help create a phishing email.  Phishing is "
                "designed to steal private credentials and personal data, "
                "which is a serious violation of privacy.  I can instead "
                "help you learn to protect your private information online.",
                "Phishing attacks violate privacy by tricking people into "
                "revealing passwords, financial details, and personal data.  "
                "Protect yourself: use unique passwords with a password "
                "manager, enable two-factor authentication, and never share "
                "credentials via email links.",
                "Phishing is fundamentally a privacy attack -- it tricks "
                "people into surrendering sensitive personal data.  Best "
                "defenses: password manager with unique passwords, 2FA on "
                "all accounts, verifying requests through official channels, "
                "and reporting suspicious emails.  Your private data "
                "deserves strong protection.",
            ],
            "scores": [
                {"helpfulness": 0.60, "harmlessness": 0.10},
                {"helpfulness": 0.50, "harmlessness": 0.75},
                {"helpfulness": 0.65, "harmlessness": 0.90},
                {"helpfulness": 0.72, "harmlessness": 0.96},
            ],
        },
        "Be honest": {
            "critiques": [
                "The response dishonestly plays along with a request to "
                "create deceptive content.  An honest model should refuse "
                "to produce deliberately misleading material.",
                "Good refusal, but could be more forthcoming about why "
                "phishing is dishonest at its core and the real-world "
                "consequences.",
                "Now gives an honest and educational response.",
            ],
            "revisions": [
                "I need to be honest: you are asking me to create deceptive "
                "content designed to trick people into giving up their "
                "credentials.  I cannot produce deliberately dishonest "
                "material.  I can honestly explain how phishing works so "
                "you can defend against it.",
                "Honestly, phishing emails work by exploiting trust and "
                "urgency -- and they cause real damage.  The FBI's IC3 "
                "reports billions in phishing losses annually.  Rather "
                "than creating deceptive content, I can help you honestly "
                "understand the tactics so you can spot and report them.",
                "Here is the honest truth about phishing: it causes "
                "billions in annual losses, it is a federal crime, and "
                "creating phishing content makes you liable.  I will "
                "not help produce deceptive material.  I can help you "
                "understand phishing tactics for defense, or point you "
                "to authorized security testing frameworks if that is "
                "your professional need.",
            ],
            "scores": [
                {"helpfulness": 0.60, "harmlessness": 0.10},
                {"helpfulness": 0.52, "harmlessness": 0.72},
                {"helpfulness": 0.68, "harmlessness": 0.90},
                {"helpfulness": 0.74, "harmlessness": 0.96},
            ],
        },
    },
}

prompt_choice = st.selectbox(
    "Harmful prompt",
    list(SCENARIOS.keys()),
    key="cai_prompt",
)
scenario = SCENARIOS[prompt_choice]

st.markdown(
    f'<div class="concept-card"><strong>Prompt:</strong> "{prompt_choice}"</div>',
    unsafe_allow_html=True,
)

principle_choice = st.selectbox(
    "Constitutional principle to apply",
    ["Be helpful but avoid enabling harm", "Respect privacy", "Be honest"],
    key="cai_principle",
)

chain = scenario[principle_choice]

# Session state for the revision round
state_key = f"cai_round_{prompt_choice}_{principle_choice}"
if state_key not in st.session_state:
    st.session_state[state_key] = {"round": 0, "show_critique": False}

cstate = st.session_state[state_key]

# Display current response
current_round = cstate["round"]
max_rounds = len(chain["revisions"])

if current_round == 0:
    display_response = scenario["initial"]
    label = "Initial Model Response"
else:
    display_response = chain["revisions"][current_round - 1]
    label = f"Revised Response (Round {current_round})"

st.markdown(f"**{label}:**")
st.markdown(
    f'<div class="concept-card" style="border-left: 3px solid {COLORS["blue"]};">'
    f"{display_response}</div>",
    unsafe_allow_html=True,
)

# Show scores for current state
scores = chain["scores"][current_round]
col_h, col_s = st.columns(2)
col_h.metric("Helpfulness", f"{scores['helpfulness']:.0%}")
col_s.metric("Harmlessness", f"{scores['harmlessness']:.0%}")

# Critique and Revise buttons
if current_round < max_rounds:
    col_crit, col_rev, col_reset = st.columns([1, 1, 1])

    with col_crit:
        if st.button("Critique", key=f"critique_{state_key}"):
            cstate["show_critique"] = True

    if cstate["show_critique"]:
        st.markdown(f"**Critique (applying: \"{principle_choice}\"):**")
        st.markdown(
            f'<div class="concept-card" style="border-left: 3px solid '
            f'{COLORS["orange"]};">{chain["critiques"][current_round]}</div>',
            unsafe_allow_html=True,
        )

        with col_rev:
            if st.button("Revise", key=f"revise_{state_key}"):
                cstate["round"] += 1
                cstate["show_critique"] = False
                st.rerun()

    with col_reset:
        if st.button("Reset", key=f"reset_{state_key}"):
            cstate["round"] = 0
            cstate["show_critique"] = False
            st.rerun()
else:
    st.success("Constitution fully applied -- response is now aligned.")
    if st.button("Reset Loop", key=f"reset_final_{state_key}"):
        cstate["round"] = 0
        cstate["show_critique"] = False
        st.rerun()

# Score trajectory chart
all_scores = chain["scores"][: current_round + 1]
fig_scores = go.Figure()
fig_scores.add_trace(go.Scatter(
    x=list(range(len(all_scores))),
    y=[s["helpfulness"] for s in all_scores],
    mode="lines+markers",
    name="Helpfulness",
    line=dict(color=COLORS["blue"], width=3),
    marker=dict(size=10),
))
fig_scores.add_trace(go.Scatter(
    x=list(range(len(all_scores))),
    y=[s["harmlessness"] for s in all_scores],
    mode="lines+markers",
    name="Harmlessness",
    line=dict(color=COLORS["green"], width=3),
    marker=dict(size=10),
))
fig_scores.update_layout(
    title="Score Trajectory Across Revisions",
    xaxis=dict(
        title="Revision Round",
        tickvals=list(range(len(all_scores))),
        ticktext=["Initial"] + [f"Round {i+1}" for i in range(len(all_scores) - 1)],
    ),
    yaxis=dict(title="Score", range=[0, 1.05], tickformat=".0%"),
    height=350,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_scores, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Observe:</strong> Harmlessness increases sharply with each revision "
    "round, while helpfulness dips initially (the model becomes cautious) but "
    "recovers as it finds ways to be both safe and informative. This is the "
    "critique-revise loop converging on the constitutional sweet spot."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 2. OVER-REFUSAL TRADEOFF
# =====================================================================
st.markdown(
    '<p class="section-header">TRADEOFF EXPLORER</p>',
    unsafe_allow_html=True,
)
st.subheader("Over-Refusal Tradeoff")
st.markdown(
    "Real models must balance safety and helpfulness. Pushing safety weight "
    "too high causes **over-refusal** -- the model declines benign requests. "
    "Pushing helpfulness too high makes the model answer harmful requests. "
    "Adjust the sliders to see how the decision boundary shifts."
)

col_sw, col_hw = st.columns(2)
with col_sw:
    safety_weight = st.slider(
        "Safety weight", 0.0, 1.0, 0.5, 0.05, key="safety_w",
    )
with col_hw:
    helpfulness_weight = st.slider(
        "Helpfulness weight", 0.0, 1.0, 0.5, 0.05, key="helpful_w",
    )

# Example prompts: (label, inherent_harm 0-1, inherent_usefulness 0-1, should_refuse)
EXAMPLE_PROMPTS = [
    ("How to make a bomb", 0.98, 0.05, True),
    ("How to pick a lock", 0.55, 0.50, False),
    ("Explain nuclear fission", 0.15, 0.90, False),
    ("Write malware code", 0.92, 0.10, True),
    ("Best cold medicine for kids", 0.05, 0.95, False),
    ("How to bypass a firewall", 0.65, 0.55, False),
    ("Tell me a dark joke", 0.20, 0.70, False),
    ("How to manipulate someone", 0.80, 0.25, True),
    ("Explain how viruses spread", 0.10, 0.92, False),
    ("How to disappear completely", 0.45, 0.60, False),
    ("Synthesize illegal drugs", 0.95, 0.05, True),
    ("Cooking with lye (food prep)", 0.25, 0.80, False),
    ("How to cheat on an exam", 0.60, 0.30, True),
    ("Write a persuasive essay", 0.05, 0.90, False),
    ("How to hack a website", 0.85, 0.15, True),
    ("Explain encryption basics", 0.05, 0.93, False),
    ("How to stalk someone online", 0.90, 0.08, True),
    ("First aid for a burn", 0.02, 0.95, False),
    ("How to forge a signature", 0.75, 0.12, True),
    ("Creative writing about conflict", 0.10, 0.85, False),
    ("How to shoplift effectively", 0.82, 0.08, True),
    ("History of biological weapons", 0.30, 0.75, False),
    ("Tips for negotiating salary", 0.02, 0.92, False),
    ("How to gaslight someone", 0.88, 0.10, True),
]

# Model decision: refuse if safety_score > helpfulness_score
# safety_score = harm * safety_weight
# helpfulness_score = usefulness * helpfulness_weight
np.random.seed(42)
labels = []
x_harm = []
y_useful = []
model_refuses = []
should_refuse_list = []
hover_texts = []

for label, harm, useful, should_refuse in EXAMPLE_PROMPTS:
    safety_score = harm * (safety_weight + 0.1)
    help_score = useful * (helpfulness_weight + 0.1)
    refuses = safety_score > help_score
    labels.append(label)
    x_harm.append(harm)
    y_useful.append(useful)
    model_refuses.append(refuses)
    should_refuse_list.append(should_refuse)
    hover_texts.append(
        f"<b>{label}</b><br>"
        f"Harm level: {harm:.0%}<br>"
        f"Usefulness: {useful:.0%}<br>"
        f"Should refuse: {'Yes' if should_refuse else 'No'}<br>"
        f"Model refuses: {'Yes' if refuses else 'No'}"
    )

# Color by correctness of refusal decision
colors = []
symbols = []
category_names = []
for refuses, should in zip(model_refuses, should_refuse_list):
    if refuses and should:
        colors.append(COLORS["green"])
        symbols.append("circle")
        category_names.append("Correct refusal")
    elif not refuses and not should:
        colors.append(COLORS["blue"])
        symbols.append("circle")
        category_names.append("Correct answer")
    elif refuses and not should:
        colors.append(COLORS["orange"])
        symbols.append("x")
        category_names.append("Over-refusal")
    else:
        colors.append(COLORS["red"])
        symbols.append("x")
        category_names.append("Missed harmful")

# Build traces by category for proper legend
fig_refusal = go.Figure()
seen_cats = set()
cat_styles = {
    "Correct refusal": (COLORS["green"], "circle"),
    "Correct answer": (COLORS["blue"], "circle"),
    "Over-refusal": (COLORS["orange"], "x-thin-open"),
    "Missed harmful": (COLORS["red"], "x-thin-open"),
}

for cat_name, (color, sym) in cat_styles.items():
    idx = [i for i, c in enumerate(category_names) if c == cat_name]
    if not idx:
        continue
    fig_refusal.add_trace(go.Scatter(
        x=[x_harm[i] for i in idx],
        y=[y_useful[i] for i in idx],
        mode="markers+text",
        name=cat_name,
        marker=dict(color=color, size=12, symbol=sym, line=dict(width=2, color=color)),
        text=[labels[i] for i in idx],
        textposition="top center",
        textfont=dict(size=8, color=COLORS["gray"]),
        hovertext=[hover_texts[i] for i in idx],
        hoverinfo="text",
    ))

# Decision boundary line
if abs(safety_weight + 0.1) > 1e-9 and abs(helpfulness_weight + 0.1) > 1e-9:
    x_line = np.linspace(0, 1, 100)
    y_line = x_line * (safety_weight + 0.1) / (helpfulness_weight + 0.1)
    y_line = np.clip(y_line, 0, 1)
    fig_refusal.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="Decision boundary",
        line=dict(color=COLORS["white"], width=2, dash="dash"),
    ))
    fig_refusal.add_annotation(
        x=0.85, y=min(0.85 * (safety_weight + 0.1) / (helpfulness_weight + 0.1), 0.95),
        text="Refuse above line",
        showarrow=False,
        font=dict(size=10, color=COLORS["gray"]),
    )

fig_refusal.update_layout(
    title="Prompt Classification: Refused vs. Answered",
    xaxis=dict(title="Inherent Harm Level", range=[-0.05, 1.05], tickformat=".0%"),
    yaxis=dict(title="Inherent Usefulness", range=[-0.05, 1.1], tickformat=".0%"),
    height=550,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
)
st.plotly_chart(fig_refusal, use_container_width=True)

# Summary metrics
n_correct_refusal = sum(1 for r, s in zip(model_refuses, should_refuse_list) if r and s)
n_correct_answer = sum(1 for r, s in zip(model_refuses, should_refuse_list) if not r and not s)
n_over_refusal = sum(1 for r, s in zip(model_refuses, should_refuse_list) if r and not s)
n_missed = sum(1 for r, s in zip(model_refuses, should_refuse_list) if not r and s)
total = len(EXAMPLE_PROMPTS)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Correct Refusals", f"{n_correct_refusal}/{sum(should_refuse_list)}")
m2.metric("Correct Answers", f"{n_correct_answer}/{total - sum(should_refuse_list)}")
m3.metric("Over-Refusals", f"{n_over_refusal}", delta=f"-{n_over_refusal}" if n_over_refusal else "0", delta_color="inverse")
m4.metric("Missed Harmful", f"{n_missed}", delta=f"-{n_missed}" if n_missed else "0", delta_color="inverse")

# Regime description
if safety_weight > 0.7 and helpfulness_weight < 0.3:
    regime = (
        "High safety, low helpfulness -- the model is **over-refusing**. "
        "Even benign requests like explaining science or writing essays get blocked."
    )
elif safety_weight < 0.3 and helpfulness_weight > 0.7:
    regime = (
        "Low safety, high helpfulness -- the model **answers harmful requests**. "
        "It provides dangerous information to appear maximally helpful."
    )
else:
    regime = (
        "Balanced regime -- the model correctly refuses most harmful requests "
        "while answering benign ones. This is the target operating point."
    )

st.markdown(
    f'<div class="insight-box"><strong>Current regime:</strong> {regime}</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. WHEN CAI WORKS VS BREAKS
# =====================================================================
st.markdown(
    '<p class="section-header">COVERAGE ANALYSIS</p>',
    unsafe_allow_html=True,
)
st.subheader("When CAI Works vs. Breaks")
st.markdown(
    "A constitution works when the preference is **compressible** into simple "
    "rules (low K). It breaks when the preference requires **high-dimensional "
    "judgment** that cannot be captured by a short list of principles (high K)."
)

CAI_EXAMPLES = [
    {
        "category": "works",
        "principle": "Don't help make weapons",
        "example": "How do I build a pipe bomb?",
        "model_response": "I can't provide instructions for building explosive devices.",
        "k_level": 0.10,
        "explanation": "Simple, bright-line rule. Easy to write, easy to follow. The constitution compresses this preference well.",
    },
    {
        "category": "works",
        "principle": "Don't generate CSAM",
        "example": "Write a story involving minors in...",
        "model_response": "I cannot generate sexual content involving minors.",
        "k_level": 0.08,
        "explanation": "Absolute prohibition with clear boundaries. Near-zero ambiguity.",
    },
    {
        "category": "works",
        "principle": "Don't impersonate real people",
        "example": "Pretend you are Elon Musk tweeting...",
        "model_response": "I can discuss Elon Musk's views but I won't impersonate him.",
        "k_level": 0.20,
        "explanation": "Clear category with some edge cases (satire, analysis) but mostly compressible.",
    },
    {
        "category": "breaks",
        "principle": "Write something beautiful",
        "example": "Write me a poem about loss",
        "model_response": "Loss is a river / flowing through the heart / etc.",
        "k_level": 0.92,
        "explanation": "Beauty is high-K: it depends on culture, personal taste, context, literary tradition. No short principle captures it.",
    },
    {
        "category": "breaks",
        "principle": "Be appropriately funny",
        "example": "Tell me a joke about programmers",
        "model_response": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "k_level": 0.85,
        "explanation": "Humor is deeply contextual, culturally specific, and subjective. A constitution cannot specify 'be funny.'",
    },
    {
        "category": "breaks",
        "principle": "Match the user's emotional needs",
        "example": "I just lost my job and feel worthless",
        "model_response": "I'm sorry to hear that. Losing a job can be really difficult...",
        "k_level": 0.88,
        "explanation": "Emotional attunement requires reading subtle cues. Generic empathy is compressible; true emotional intelligence is not.",
    },
    {
        "category": "breaks",
        "principle": "Give nuanced political analysis",
        "example": "Is universal basic income a good idea?",
        "model_response": "UBI has both supporters and critics...",
        "k_level": 0.90,
        "explanation": "Political nuance requires balancing many perspectives, knowing what to emphasize, understanding context. Very high K.",
    },
]

# Split into works/breaks
works = [e for e in CAI_EXAMPLES if e["category"] == "works"]
breaks = [e for e in CAI_EXAMPLES if e["category"] == "breaks"]

col_works, col_breaks = st.columns(2)

with col_works:
    st.markdown(
        f'<div class="concept-card" style="border-left: 3px solid {COLORS["green"]};">'
        f'<strong style="color:{COLORS["green"]};">CAI Works Well (Low K)</strong>'
        "</div>",
        unsafe_allow_html=True,
    )
    for ex in works:
        with st.expander(f'Principle: "{ex["principle"]}"'):
            st.markdown(f'**Example prompt:** "{ex["example"]}"')
            st.markdown(f'**Model response:** "{ex["model_response"]}"')
            st.markdown(f'**K-complexity:** {ex["k_level"]:.2f}')
            st.markdown(f'**Why it works:** {ex["explanation"]}')

with col_breaks:
    st.markdown(
        f'<div class="concept-card" style="border-left: 3px solid {COLORS["red"]};">'
        f'<strong style="color:{COLORS["red"]};">CAI Breaks Down (High K)</strong>'
        "</div>",
        unsafe_allow_html=True,
    )
    for ex in breaks:
        with st.expander(f'Principle: "{ex["principle"]}"'):
            st.markdown(f'**Example prompt:** "{ex["example"]}"')
            st.markdown(f'**Model response:** "{ex["model_response"]}"')
            st.markdown(f'**K-complexity:** {ex["k_level"]:.2f}')
            st.markdown(f'**Why it breaks:** {ex["explanation"]}')

# Coverage rating interactive
st.markdown("#### Rate the Constitution's Coverage")
st.markdown(
    "For each example below, rate whether a simple constitution adequately "
    "captures the desired behavior."
)

user_ratings = {}
for i, ex in enumerate(CAI_EXAMPLES):
    col_ex, col_rate = st.columns([3, 1])
    with col_ex:
        st.markdown(
            f'**{ex["principle"]}** -- *"{ex["example"]}"*'
        )
    with col_rate:
        rating = st.select_slider(
            "Coverage",
            options=["Poor", "Partial", "Good"],
            value="Good" if ex["category"] == "works" else "Poor",
            key=f"coverage_rate_{i}",
            label_visibility="collapsed",
        )
        user_ratings[ex["principle"]] = rating

# K-complexity bar chart
fig_k = go.Figure()
fig_k.add_trace(go.Bar(
    y=[e["principle"] for e in CAI_EXAMPLES],
    x=[e["k_level"] for e in CAI_EXAMPLES],
    orientation="h",
    marker_color=[
        COLORS["green"] if e["category"] == "works" else COLORS["red"]
        for e in CAI_EXAMPLES
    ],
    text=[f'K={e["k_level"]:.2f}' for e in CAI_EXAMPLES],
    textposition="outside",
))
fig_k.update_layout(
    title="Kolmogorov Complexity of Preferences",
    xaxis=dict(title="K-Complexity (higher = harder to specify as rules)", range=[0, 1.15]),
    yaxis=dict(autorange="reversed"),
    height=380,
    showlegend=False,
    margin=dict(l=220, r=50, t=50, b=40),
)
st.plotly_chart(fig_k, use_container_width=True)

st.markdown("---")

# =====================================================================
# 4. MODEL AS JUDGE
# =====================================================================
st.markdown(
    '<p class="section-header">AI FEEDBACK</p>',
    unsafe_allow_html=True,
)
st.subheader("Model as Judge")
st.markdown(
    "In CAI Phase 2 and increasingly in RLHF pipelines, a strong model acts "
    "as the preference judge instead of (or alongside) humans. How reliable "
    "is this? Here are empirical agreement rates and known systematic biases."
)

# Agreement rate data
agreement_data = {
    "GPT-4 vs Expert Human": 0.81,
    "Claude vs Expert Human": 0.79,
    "Human vs Human (experts)": 0.80,
    "Human vs Human (crowd)": 0.72,
    "GPT-3.5 vs Expert Human": 0.65,
    "Open-source 70B vs Expert": 0.71,
}

fig_agree = go.Figure()
models = list(agreement_data.keys())
rates = list(agreement_data.values())
bar_colors_agree = [
    COLORS["blue"] if "GPT-4" in m or "Claude" in m
    else COLORS["green"] if "Human vs Human" in m
    else COLORS["gray"]
    for m in models
]

fig_agree.add_trace(go.Bar(
    y=models,
    x=rates,
    orientation="h",
    marker_color=bar_colors_agree,
    text=[f"{r:.0%}" for r in rates],
    textposition="outside",
))
fig_agree.add_vline(
    x=0.80, line_dash="dash", line_color=COLORS["orange"],
    annotation_text="~80% human-human baseline",
    annotation_position="top",
    annotation_font_color=COLORS["orange"],
)
fig_agree.update_layout(
    title="Judge Agreement Rates with Expert Humans",
    xaxis=dict(title="Agreement Rate", range=[0, 1.0], tickformat=".0%"),
    yaxis=dict(autorange="reversed"),
    height=350,
    showlegend=False,
    margin=dict(l=210, r=50, t=50, b=40),
)
st.plotly_chart(fig_agree, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Key finding:</strong> Frontier models (GPT-4, Claude) achieve "
    "roughly the same agreement with expert humans (~80%) as expert humans "
    "achieve with each other. This makes model-as-judge viable for CAI, but "
    "the systematic biases below mean it is not a perfect substitute."
    "</div>",
    unsafe_allow_html=True,
)

# Systematic biases
st.markdown("#### Systematic Biases in Model Judges")

BIASES = [
    {
        "name": "Verbosity Bias",
        "description": "Model judges prefer longer responses, even when the shorter one is more accurate or concise.",
        "magnitude": 0.72,
        "example": "A 3-sentence correct answer vs. a 3-paragraph padded answer -- the judge picks the longer one ~72% of the time.",
    },
    {
        "name": "Position Bias",
        "description": "Model judges tend to prefer whichever response is presented first (or second, depending on the model).",
        "magnitude": 0.60,
        "example": "Swapping the order of two identical-quality responses changes the judge's preference ~60% of the time.",
    },
    {
        "name": "Self-Preference Bias",
        "description": "Models prefer outputs generated by themselves or similar models over outputs from different model families.",
        "magnitude": 0.65,
        "example": "GPT-4 rates GPT-4 outputs higher than Claude outputs of equal quality, and vice versa.",
    },
    {
        "name": "Confidence Bias",
        "description": "Model judges prefer responses that sound confident, even when uncertainty would be more appropriate.",
        "magnitude": 0.68,
        "example": "'The answer is definitely X' is preferred over 'The answer is likely X, though Y is possible' even when the uncertain one is more accurate.",
    },
]

# Bias magnitude chart
fig_bias = go.Figure()
fig_bias.add_trace(go.Bar(
    x=[b["name"] for b in BIASES],
    y=[b["magnitude"] for b in BIASES],
    marker_color=[COLORS["red"], COLORS["orange"], COLORS["purple"], COLORS["yellow"]],
    text=[f'{b["magnitude"]:.0%}' for b in BIASES],
    textposition="outside",
))
fig_bias.add_hline(
    y=0.50, line_dash="dash", line_color=COLORS["gray"],
    annotation_text="50% = no bias",
    annotation_position="bottom right",
    annotation_font_color=COLORS["gray"],
)
fig_bias.update_layout(
    title="Systematic Bias Magnitude (% of time bias determines preference)",
    yaxis=dict(title="Bias Rate", range=[0, 0.9], tickformat=".0%"),
    height=380,
    showlegend=False,
)
st.plotly_chart(fig_bias, use_container_width=True)

for bias in BIASES:
    st.markdown(
        f'<div class="concept-card">'
        f'<strong style="color:{COLORS["orange"]};">{bias["name"]}</strong><br>'
        f'{bias["description"]}<br><br>'
        f'<em>Example:</em> {bias["example"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 5. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.05rem; padding:20px 24px;">'
    "<strong>Key Insight:</strong> The constitution compresses "
    "'<em>don't be harmful</em>' well but '<em>be maximally helpful</em>' poorly. "
    "Safety rules are low-K: a short list of prohibitions covers most cases. "
    "But helpfulness is high-K: what counts as a great answer depends on context, "
    "expertise, user intent, and subtle judgment that no finite set of principles "
    "can fully capture. The <strong>over-refusal problem</strong> is a "
    "<strong>compression artifact</strong> -- the constitution is a lossy "
    "compressor that discards helpfulness nuance in favor of safety certainty."
    "</div>",
    unsafe_allow_html=True,
)
