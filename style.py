import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# ── Shared dark Plotly template ──────────────────────────────────────────────
COLORS = {
    "blue": "#4A90D9",
    "red": "#E74C3C",
    "green": "#2ECC71",
    "orange": "#F39C12",
    "purple": "#9B59B6",
    "cyan": "#1ABC9C",
    "pink": "#E91E8A",
    "yellow": "#F1C40F",
    "gray": "#95A5A6",
    "white": "#ECF0F1",
    "bg": "#0E1117",
    "card": "#1E2130",
    "grid": "#2A2D3E",
}

DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["white"], family="Inter, system-ui, sans-serif"),
        title=dict(font=dict(size=18)),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickfont=dict(size=12),
        ),
        colorway=[
            COLORS["blue"], COLORS["red"], COLORS["green"],
            COLORS["orange"], COLORS["purple"], COLORS["cyan"],
            COLORS["pink"], COLORS["yellow"],
        ],
        margin=dict(l=50, r=30, t=50, b=40),
    )
)
pio.templates["dark_custom"] = DARK_TEMPLATE
pio.templates.default = "dark_custom"


# ── CSS injection ────────────────────────────────────────────────────────────
def inject_custom_css():
    import streamlit as st
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp { font-family: 'Inter', system-ui, sans-serif; }

    .big-formula {
        background: #1E2130;
        border-radius: 10px;
        padding: 18px 24px;
        font-family: 'Courier New', monospace;
        font-size: 1.15rem;
        color: #ECF0F1;
        margin: 12px 0;
        border-left: 4px solid #4A90D9;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        border-left: 4px solid #F39C12;
        color: #ECF0F1;
    }

    .insight-box strong { color: #F39C12; }

    .concept-card {
        background: #1E2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #2A2D3E;
    }

    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #95A5A6;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Utility functions ────────────────────────────────────────────────────────
def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    x_scaled = x / max(temperature, 1e-10)
    x_scaled -= x_scaled.max()
    e = np.exp(x_scaled)
    return e / e.sum()


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log2(p))
