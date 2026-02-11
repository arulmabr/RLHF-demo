"""
Page 1 -- The Big Idea: Prediction is Compression
Compression requires understanding.
"""

from style import inject_custom_css, COLORS, softmax, entropy
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import zlib
import string

# ── Page Setup ───────────────────────────────────────────────────────────────
inject_custom_css()

st.markdown(
    '<p class="section-header">SECTION I</p>',
    unsafe_allow_html=True,
)
st.title("The Big Idea: Prediction is Compression")

st.markdown(
    """
    The single deepest idea in modern AI is deceptively simple:

    > **A model that predicts well must compress well, and compression requires understanding.**

    Every regularity a language model discovers -- grammar, facts, logic, style --
    lets it assign shorter codes to likely continuations and longer codes to
    surprises.  The theoretical floor is **Kolmogorov complexity**:
    """
)

st.markdown(
    '<div class="big-formula">'
    "K(x) = length of the shortest program that produces x"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    A string with lots of structure (e.g. `"abcabcabc..."`) has low K -- a tiny
    program can regenerate it.  A string of truly random bytes has K roughly equal
    to its own length -- there is no shorter description.

    Language models chase the same goal: **minimize the expected code length of
    the next token**, which is exactly the cross-entropy loss.
    """
)

st.markdown("---")

# =====================================================================
# 1. COMPRESSION GAME DEMO
# =====================================================================
st.markdown(
    '<p class="section-header">INTERACTIVE DEMO</p>',
    unsafe_allow_html=True,
)
st.subheader("Compression Game")
st.markdown(
    "Type (or paste) any text below and watch how well `zlib` can compress it. "
    "Try three experiments: **(a)** highly repetitive text, **(b)** normal "
    "English prose, **(c)** random gibberish."
)

preset_options = {
    "-- custom --": "",
    "Repetitive": "abcabc" * 80,
    "English prose": (
        "The quick brown fox jumps over the lazy dog. "
        "Language models learn to predict the next token in a sequence. "
        "Prediction is equivalent to compression because a good predictor "
        "assigns short codes to likely events and long codes to unlikely ones. "
        "This deep connection between prediction and compression is the "
        "foundation of modern large language model training."
    ),
    "Random characters": "".join(
        np.random.RandomState(42).choice(
            list(string.ascii_letters + string.digits + string.punctuation), 300
        )
    ),
}

preset = st.selectbox("Quick presets", list(preset_options.keys()))

default_text = preset_options[preset] if preset != "-- custom --" else (
    "Try typing something here..."
)

user_text = st.text_area(
    "Your text",
    value=default_text,
    height=120,
    key="compression_input",
)

if user_text:
    raw_bytes = user_text.encode("utf-8")
    compressed_bytes = zlib.compress(raw_bytes, level=9)
    original_size = len(raw_bytes)
    compressed_size = len(compressed_bytes)
    ratio = compressed_size / max(original_size, 1)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Original size", f"{original_size} bytes")
    col_m2.metric("Compressed size", f"{compressed_size} bytes")
    col_m3.metric(
        "Compression ratio",
        f"{ratio:.2%}",
        delta=f"-{(1 - ratio) * 100:.1f}% smaller" if ratio < 1 else None,
        delta_color="normal",
    )

    fig_comp = go.Figure(
        data=[
            go.Bar(
                x=["Original", "Compressed"],
                y=[original_size, compressed_size],
                marker_color=[COLORS["blue"], COLORS["green"]],
                text=[f"{original_size} B", f"{compressed_size} B"],
                textposition="outside",
            )
        ]
    )
    fig_comp.update_layout(
        title="Original vs Compressed Size",
        yaxis_title="Bytes",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<strong>Why does this matter?</strong> Repetitive text compresses "
        "dramatically because a short program (copy this pattern N times) "
        "suffices. Random text barely compresses at all -- every byte is a "
        "surprise. A language model's cross-entropy loss is a direct measure "
        "of how compressible the text is under that model's predictions."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 2. PREDICTABILITY HEATMAP
# =====================================================================
st.markdown(
    '<p class="section-header">VISUALIZATION</p>',
    unsafe_allow_html=True,
)
st.subheader("Character Predictability Heatmap")
st.markdown(
    "Below we color each character by how *predictable* it is given the "
    "previous character (a simple bigram model). **Dark blue = highly "
    "predictable**, **bright yellow = surprising**. Notice how spaces after "
    "common words and repeated patterns are very predictable."
)

heatmap_sentences = {
    "The quick brown fox jumps over the lazy dog.": "Classic pangram",
    "aaaaaabbbbbbcccccc": "Repetitive characters",
    "To be or not to be, that is the question.": "Shakespeare",
    "xQz7!pL@3kW*9mNv": "Random-looking string",
}

selected_sentence = st.selectbox(
    "Choose a sentence",
    list(heatmap_sentences.keys()),
    format_func=lambda s: f"{heatmap_sentences[s]}: \"{s}\"",
)


def bigram_surprise(text: str) -> np.ndarray:
    """
    Estimate per-character surprise using a simple English bigram model.
    Returns values in [0, 1] where 0 = very predictable, 1 = very surprising.
    """
    # Build a rough bigram frequency table from common English patterns.
    # We use a smoothed count approach: frequent bigrams get low surprise.
    common_bigrams = (
        "th he in er an re on at en nd ti es or te of ed is it al ar st "
        "to nt ng se ha as ou io le ve co me de hi ri ro ic ne ea ra ce "
        "li ch ll be ma si om ur "
        # Space-letter bigrams (common word starts after space)
        "e t a o i n s h r"
    ).split()

    bigram_set = set(common_bigrams)

    scores = np.ones(len(text), dtype=np.float64) * 0.5  # default mid-surprise

    for i, ch in enumerate(text):
        # Character-level heuristics
        if ch == " ":
            # Spaces are very predictable after punctuation or long words
            scores[i] = 0.1
        elif i > 0:
            bigram = (text[i - 1] + ch).lower()
            if bigram in bigram_set:
                scores[i] = 0.15  # common bigram -> very predictable
            elif text[i - 1] == text[i]:
                scores[i] = 0.2  # repeated character
            elif ch.lower() in "etaoinshrdlcumwfgypbvkjxqz":
                # Scale by letter frequency (rough ordering)
                freq_rank = "etaoinshrdlcumwfgypbvkjxqz".index(ch.lower())
                scores[i] = 0.25 + 0.025 * freq_rank
            else:
                scores[i] = 0.85  # unusual character

        # Punctuation and digits are generally more surprising
        if ch in string.punctuation:
            scores[i] = max(scores[i], 0.7)
        if ch in string.digits:
            scores[i] = max(scores[i], 0.75)

    return scores


surprise_vals = bigram_surprise(selected_sentence)

# Build heatmap: one row, each column is a character
chars = list(selected_sentence)
# Wrap into rows of a max width for readability
MAX_ROW_LEN = 40
rows_chars = [
    chars[i : i + MAX_ROW_LEN] for i in range(0, len(chars), MAX_ROW_LEN)
]
rows_vals = [
    surprise_vals[i : i + MAX_ROW_LEN] for i in range(0, len(surprise_vals), MAX_ROW_LEN)
]

# Pad last row so the matrix is rectangular
if len(rows_chars) > 1 and len(rows_chars[-1]) < MAX_ROW_LEN:
    pad = MAX_ROW_LEN - len(rows_chars[-1])
    rows_chars[-1] += [""] * pad
    rows_vals[-1] = np.concatenate([rows_vals[-1], np.full(pad, np.nan)])

z_matrix = np.array([r if isinstance(r, np.ndarray) else np.array(r) for r in rows_vals])
text_matrix = rows_chars

# Row labels (just row indices, reversed so first row is on top)
y_labels = [f"Row {i + 1}" for i in range(len(rows_chars))]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=z_matrix,
        text=text_matrix,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=13, color="white"),
        colorscale=[
            [0.0, "#1a237e"],   # very predictable  -> dark blue
            [0.3, "#1565c0"],
            [0.5, "#43a047"],   # medium             -> green
            [0.7, "#f9a825"],
            [1.0, "#ff6f00"],   # very surprising    -> bright orange
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Surprise",
            tickvals=[0, 0.5, 1],
            ticktext=["Predictable", "Medium", "Surprising"],
        ),
        hovertemplate='Char: %{text}<br>Surprise: %{z:.2f}<extra></extra>',
    )
)
fig_heat.update_layout(
    title="Per-Character Surprise (Bigram Model)",
    height=60 + 60 * len(rows_chars),
    yaxis=dict(
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
        autorange="reversed",
    ),
    xaxis=dict(showticklabels=False),
    margin=dict(l=70, r=30, t=50, b=20),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<strong>Reading the heatmap:</strong> Characters that follow common "
    "English bigrams (like 'th', 'he', 'in') light up as predictable. "
    "Punctuation, rare letters, and random characters show high surprise. "
    "A language model with a richer context window would push even more "
    "characters toward predictable -- that is the power of compression."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# 3. KOLMOGOROV COMPLEXITY EXPLORER
# =====================================================================
st.markdown(
    '<p class="section-header">EXPLORER</p>',
    unsafe_allow_html=True,
)
st.subheader("Kolmogorov Complexity and Emergent Capabilities")
st.markdown(
    "Not all capabilities are equally hard to learn. Capabilities with **low "
    "Kolmogorov complexity** (short description length) are acquired first "
    "during pre-training. Those with **high K** resist compression and emerge "
    "only at scale -- or require post-training."
)

# Capability definitions: (name, approximate_K, category)
capabilities = [
    # Low K -- easy to compress / learn
    ("Syntax & Grammar", 0.10, "low"),
    ("Spelling Rules", 0.15, "low"),
    ("Math (arithmetic)", 0.20, "low"),
    ("Formal Logic", 0.25, "low"),
    ("Code Patterns", 0.30, "low"),
    ("Factual Knowledge", 0.40, "low"),
    # High K -- hard to compress / learn
    ("Pragmatics", 0.55, "high"),
    ("Cultural Context", 0.65, "high"),
    ("Emotional Nuance", 0.75, "high"),
    ("Creative Taste", 0.82, "high"),
    ("Individual Voice", 0.90, "high"),
    ("Value Alignment", 0.95, "high"),
]

model_capacity = st.slider(
    "Model capacity (scaling pre-training compute)",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.01,
    format="%.0f%%",
    help="Slide right to simulate larger pre-training runs.",
)


def capability_mastery(k: float, capacity: float, steepness: float = 18.0) -> float:
    """Sigmoid mastery curve: low-K capabilities are learned first."""
    return float(1.0 / (1.0 + np.exp(steepness * (k - capacity))))


mastery = [capability_mastery(k, model_capacity) for _, k, _ in capabilities]
names = [n for n, _, _ in capabilities]
k_vals = [k for _, k, _ in capabilities]
cats = [c for _, _, c in capabilities]

bar_colors = [
    COLORS["green"] if m > 0.6 else (COLORS["orange"] if m > 0.25 else COLORS["red"])
    for m in mastery
]

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#2ECC71;">Easy to Compress (Low K)</strong>'
        "<ul>"
        + "".join(
            f"<li>{n}</li>" for n, _, c in capabilities if c == "low"
        )
        + "</ul></div>",
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        '<div class="concept-card">'
        '<strong style="color:#E74C3C;">Hard to Compress (High K)</strong>'
        "<ul>"
        + "".join(
            f"<li>{n}</li>" for n, _, c in capabilities if c == "high"
        )
        + "</ul></div>",
        unsafe_allow_html=True,
    )

fig_cap = go.Figure(
    data=[
        go.Bar(
            y=names,
            x=mastery,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{m:.0%}" for m in mastery],
            textposition="outside",
        )
    ]
)
fig_cap.update_layout(
    title=f"Capability Mastery at {model_capacity:.0%} Capacity",
    xaxis=dict(title="Mastery", range=[0, 1.15], tickformat=".0%"),
    yaxis=dict(autorange="reversed"),
    height=460,
    showlegend=False,
    margin=dict(l=140, r=40, t=50, b=40),
)
st.plotly_chart(fig_cap, use_container_width=True)

st.markdown(
    "Drag the slider to the right and watch low-K capabilities saturate "
    "first while high-K capabilities lag behind. This is the **emergence "
    "curve**: structure that is easy to compress is learned early."
)

st.markdown("---")

# =====================================================================
# 4. KEY INSIGHT
# =====================================================================
st.markdown(
    '<div class="insight-box" style="font-size:1.05rem; padding:20px 24px;">'
    "<strong>Key Insight:</strong> Models learn the left column (syntax, "
    "logic, facts) first because these have low Kolmogorov complexity -- "
    "they are highly compressible. The right column (emotional nuance, "
    "creative taste, individual voice, value alignment) resists compression. "
    "That is precisely where <em>post-training</em> becomes essential."
    "</div>",
    unsafe_allow_html=True,
)
