from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.environment.pricing_economics import calculate_pricing_outcome

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.environment.counterfactual_simulator import CounterfactualSimulator
from src.models.ppo_agent import PPOAgent


# --------------------------------------------------
# Paths and constants
# --------------------------------------------------

PRODUCTS_PATH = ROOT / "data/processed/products_clean.parquet"
STATES_PATH = ROOT / "data/processed/state/state.npy"
INDICES_PATH = ROOT / "data/processed/state/state_row_indices.npy"

PPO_PATH = (
    ROOT
    / "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth"
)
PREDICTOR_DIR = ROOT / "models/sales_predictor"

STATE_DIM = 111
ACTION_DIM = 8
SUPPORTED_DISCOUNTS = [5, 10, 15, 20, 25, 30, 35, 40]
ALL_DISCOUNTS = [0, *SUPPORTED_DISCOUNTS]

# Fixed to the paper's balanced setting
ALPHA = 0.5
BETA = 0.5


# --------------------------------------------------
# Palette (Dark Luxe)
# --------------------------------------------------

BG = "#1A1F2E"          # Deep navy-charcoal
SURFACE = "#232839"     # Card surface
TEXT = "#EEEDEB"        # Silk (inverted)
MUTED = "#9CA0AE"       # Muted grey-blue
ACCENT = "#B89B7A"      # Champagne
BORDER = "#2F3548"      # Faint border


# --------------------------------------------------
# Page configuration and CSS
# --------------------------------------------------

st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="•",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {TEXT};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-weight: 600;
    }}
    .stApp p, .stApp label, .stApp span {{
        color: {TEXT};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    section[data-testid="stSidebar"] {{ display: none; }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT};
        font-size: 1.75rem;
        font-weight: 600;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .stButton > button {{
        background-color: {ACCENT};
        color: #1A1815;
        border: none;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.55rem 1.2rem;
    }}
    .stButton > button:hover {{
        background-color: #D2BBA0;
        color: #1A1815;
    }}
    /* Inputs and dropdowns for dark mode */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] input,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div {{
        color: {TEXT} !important;
        background-color: {SURFACE} !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: {BORDER} !important;
    }}
    .stTextInput input::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}
    .stTextInput > div,
    .stSelectbox > div {{
        background-color: {SURFACE} !important;
    }}
    /* Dataframe overrides for dark theme */
    .stDataFrame, .stDataFrame > div {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
    }}
    .stDataFrame table tbody tr td {{
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    .stDataFrame table thead tr th {{
        color: {MUTED} !important;
        background-color: {SURFACE} !important;
        border-color: {BORDER} !important;
    }}
    /* Captions */
    div[data-testid="stCaptionContainer"] p {{ color: {MUTED}; }}
    /* Hide Streamlit chrome */
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Backend
# --------------------------------------------------

@st.cache_data
def load_product_data():
    products = pd.read_parquet(PRODUCTS_PATH)
    states = np.load(STATES_PATH, allow_pickle=False)
    source_indices = np.load(INDICES_PATH, allow_pickle=False)

    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected state shape (N, {STATE_DIM}).")
    if len(states) != len(source_indices):
        raise ValueError("State rows and source-index mapping do not match.")
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("Duplicate source indices found in state mapping.")
    if np.any(source_indices < 0) or np.any(source_indices >= len(products)):
        raise ValueError("State mapping contains invalid product row indices.")

    position_by_row = {int(r): p for p, r in enumerate(source_indices)}
    return products, states, position_by_row


@st.cache_resource
def load_models():
    if not PPO_PATH.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {PPO_PATH}")

    simulator = CounterfactualSimulator(
        checkpoint_dir=str(PREDICTOR_DIR),
        device=torch.device("cpu"),
    )

    checkpoint = torch.load(PPO_PATH, map_location="cpu", weights_only=False)
    saved_action_dim = checkpoint.get("action_dim")
    if saved_action_dim is not None and int(saved_action_dim) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} PPO actions, got {saved_action_dim}.")

    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()

    for p in agent.actor.parameters():
        p.requires_grad_(False)
    for p in agent.critic.parameters():
        p.requires_grad_(False)

    return simulator, agent


def format_number(value, decimals=2):
    try:
        number = float(value)
        return f"{number:,.{decimals}f}" if np.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def make_product_label(row_index, row):
    asin = str(row.get("ASIN", "Unknown ASIN"))
    title = str(row.get("Title", "Untitled product"))
    if len(title) > 85:
        title = title[:82] + "..."
    return f"{asin} · {title}"


def compute_bin_metrics(product: pd.Series, predictions: dict[int, float]) -> pd.DataFrame:
    observed_price = float(product.get("Buybox Price (USD)"))
    observed_discount = float(product.get("Discount"))
    last_sales = float(product.get("Last Monthly Sales"))
    gmr = float(product.get("Gross Margin Rate"))

    baseline = calculate_pricing_outcome(
        observed_price=observed_price,
        observed_discount=observed_discount,
        last_monthly_sales=last_sales,
        predicted_growth_rate=0.0,
        candidate_discount=observed_discount,
        gross_margin_rate=gmr,
    )
    baseline_margin = float(baseline.estimated_margin)
    baseline_sales = float(last_sales)

    rows = []
    for d in ALL_DISCOUNTS:
        growth = float(predictions[d])
        outcome = calculate_pricing_outcome(
            observed_price=observed_price,
            observed_discount=observed_discount,
            last_monthly_sales=last_sales,
            predicted_growth_rate=growth,
            candidate_discount=float(d),
            gross_margin_rate=gmr,
        )
        margin = float(outcome.estimated_margin)
        delta_m = (margin - baseline_margin) / max(abs(baseline_margin), 1e-8)
        delta_q = (outcome.estimated_sales - baseline_sales) / max(abs(baseline_sales), 1e-8)
        rows.append({
            "Discount (%)": int(d),
            "Growth (ΔQ)": delta_q,
            "Margin (ΔM)": delta_m,
        })

    return pd.DataFrame(rows)


def add_reward_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reward"] = ALPHA * df["Margin (ΔM)"] + BETA * df["Growth (ΔQ)"]
    return df


def make_why_panel(df: pd.DataFrame, chosen_discount: int) -> str:
    chosen = df[df["Discount (%)"] == chosen_discount].iloc[0]
    growth_peak = df.loc[df["Growth (ΔQ)"].idxmax()]
    reward_peak = df.loc[df["Reward"].idxmax()]

    s1 = (
        f"The policy chose <b>{int(chosen['Discount (%)'])}%</b> with a reward of "
        f"<b>{float(chosen['Reward']):+.4f}</b>."
    )

    s2 = (
        f"Growth peaks at <b>{int(growth_peak['Discount (%)'])}%</b> "
        f"(ΔQ = <b>{float(growth_peak['Growth (ΔQ)']):.4f}</b>), "
        f"but the margin at that discount is "
        f"<b>{float(growth_peak['Margin (ΔM)']):+.4f}</b> vs "
        f"<b>{float(chosen['Margin (ΔM)']):+.4f}</b> at "
        f"<b>{int(chosen['Discount (%)'])}%</b>."
    )

    if int(reward_peak["Discount (%)"]) != int(chosen["Discount (%)"]):
        s3 = (
            f"Under the balanced reward configuration (α=0.5, β=0.5), the reward at "
            f"<b>{int(reward_peak['Discount (%)'])}%</b> "
            f"(<b>{float(reward_peak['Reward']):+.4f}</b>) is higher than the reward at "
            f"<b>{int(chosen['Discount (%)'])}%</b> "
            f"(<b>{float(chosen['Reward']):+.4f}</b>). The PPO policy is frozen and was trained "
            f"on a specific reward signal, so it does not always exactly match the reward's argmax."
        )
    else:
        s3 = (
            f"Under the balanced reward configuration (α=0.5, β=0.5), the reward is maximized at "
            f"<b>{int(chosen['Discount (%)'])}%</b>, matching the frozen PPO choice."
        )

    return "  \n".join([s1, s2, s3])


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("<h1 style='margin-bottom:0.2rem;'>Dynamic Pricing Intelligence</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED}; margin-top:0; font-size:1.05rem;'>"
    f"Interactive pricing recommendation using the frozen 8-action PPO policy "
    f"and the frozen sales predictor."
    f"</p>",
    unsafe_allow_html=True,
)

required_paths = [PRODUCTS_PATH, STATES_PATH, INDICES_PATH, PPO_PATH, PREDICTOR_DIR / "config.json"]
missing = [str(p) for p in required_paths if not p.exists()]
if missing:
    st.error("Required project files are missing:")
    for p in missing:
        st.code(p)
    st.stop()

try:
    products, states, position_by_row = load_product_data()
    simulator, agent = load_models()
except Exception as exc:
    st.error(f"Could not initialize the pricing pipeline: {exc}")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# Panel 1 — product selector and overview
# --------------------------------------------------

st.markdown("### Select a product")

search_query = st.text_input(
    "Search by title, ASIN, or brand",
    placeholder="For example: pillow, B08Z2SG9FC, ORSEN",
    label_visibility="collapsed",
)

if search_query.strip():
    q = search_query.strip().casefold()
    mask = (
        products["Title"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        | products["ASIN"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        | products["Brand"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
    )
    matching = [int(i) for i in products.index[mask].tolist() if int(i) in position_by_row][:100]
else:
    matching = [int(i) for i in products.index[:200] if int(i) in position_by_row]

if not matching:
    st.warning("No eligible products matched your search.")
    st.stop()

labels = {i: make_product_label(i, products.loc[i]) for i in matching}
selected_row = st.selectbox(
    "Available products",
    options=matching,
    format_func=lambda i: labels[i],
    label_visibility="collapsed",
)

product = products.loc[selected_row]
state_position = position_by_row[selected_row]
state = np.asarray(states[state_position], dtype=np.float32)

if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
    st.error("The selected product has an invalid feature state.")
    st.stop()

# Product image
image_url = product.get("Main Image URL")
if pd.notna(image_url) and str(image_url).startswith(("https://", "http://")):
    try:
        st.image(str(image_url), width=190)
    except Exception:
        st.caption("Product image could not be displayed.")

# Stat cards
current_price = pd.to_numeric(pd.Series([product.get("Buybox Price (USD)")]), errors="coerce").iloc[0]
current_discount = pd.to_numeric(pd.Series([product.get("Discount")]), errors="coerce").iloc[0]
monthly_sales = pd.to_numeric(pd.Series([product.get("Last Monthly Sales")]), errors="coerce").iloc[0]
current_growth = pd.to_numeric(pd.Series([product.get("Estimated Monthly Sales Growth Rate")]), errors="coerce").iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current price", f"${current_price:,.2f}" if pd.notna(current_price) else "N/A")
c2.metric("Current discount", f"{current_discount:.0f}%" if pd.notna(current_discount) else "N/A")
c3.metric("Last monthly sales", format_number(monthly_sales, 0))
c4.metric("Recorded growth", f"{current_growth:.2f}%" if pd.notna(current_growth) else "N/A")


# --------------------------------------------------
# Panel 2 — inference and recommendation
# --------------------------------------------------

st.markdown("### Recommendation")

# Force the user to click the button to generate. 
# We check session_state to see if we already have the result for this product.
has_result = (
    "result" in st.session_state 
    and st.session_state["result"]["selected_row"] == selected_row
)

# The button only appears if we don't have a result yet for this product.
if not has_result:
    if st.button("Generate recommendation", type="primary"):
        try:
            with st.spinner("Running the trained pricing models..."):
                predictions = simulator.predict_all_discounts(state)
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    dist = agent.actor.get_distribution(state_tensor)
                    probs = dist.probs.squeeze(0).cpu().numpy()
                    action = int(np.argmax(probs))
                    critic_value = float(agent.critic(state_tensor).item())

                recommended_discount = SUPPORTED_DISCOUNTS[action]
                recommended_growth = float(predictions[recommended_discount])

                reference_price = None
                recommended_price = None
                if (
                    pd.notna(current_price)
                    and pd.notna(current_discount)
                    and current_price > 0
                    and 0 <= current_discount < 100
                ):
                    reference_price = current_price / (1.0 - float(current_discount) / 100.0)
                    recommended_price = reference_price * (1.0 - recommended_discount / 100.0)

                st.session_state["result"] = {
                    "selected_row": selected_row,
                    "predictions": predictions,
                    "recommended_discount": recommended_discount,
                    "recommended_growth": recommended_growth,
                    "recommended_price": recommended_price,
                    "current_price": float(current_price) if pd.notna(current_price) else None,
                    "critic_value": critic_value,
                }
                # Rerun immediately so the results appear without a second click
                st.rerun()
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.exception(exc)


# --------------------------------------------------
# Panel 3 — Results (only shown if a result exists for the current product)
# --------------------------------------------------

if has_result:
    result = st.session_state["result"]
    predictions = result["predictions"]
    recommended_discount = result["recommended_discount"]
    recommended_price = result["recommended_price"]

    df = compute_bin_metrics(product, predictions)
    df = add_reward_column(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("PPO-recommended discount", f"{recommended_discount}%")
    c2.metric(
        "Calculated selling price",
        f"${recommended_price:,.2f}" if recommended_price is not None else "Unavailable",
    )
    c3.metric("Predicted growth", f"{result['recommended_growth']:+.2f}")

    st.markdown("#### Why this discount?")
    st.markdown(make_why_panel(df, recommended_discount), unsafe_allow_html=True)

    st.markdown("### Growth and reward analysis")

    growth_peak_row = df.loc[df["Growth (ΔQ)"].idxmax()]
    reward_peak_row = df.loc[df["Reward"].idxmax()]

    col_left, col_right = st.columns(2)

    # --------------------------------------------------
    # Chart A — Sales growth by discount
    # --------------------------------------------------
    with col_left:
        fig_growth = go.Figure()

        fig_growth.add_trace(go.Scatter(
            x=df["Discount (%)"], y=df["Growth (ΔQ)"],
            mode="lines+markers", name="Sales growth (ΔQ)",
            line=dict(color=TEXT, width=2.5),
            marker=dict(color=TEXT, size=7),
            hovertemplate="Discount %{x}%<br>ΔQ %{y:.4f}<extra></extra>",
        ))

        fig_growth.add_trace(go.Scatter(
            x=[int(growth_peak_row["Discount (%)"])],
            y=[float(growth_peak_row["Growth (ΔQ)"])],
            mode="markers+text", name="Growth peak",
            marker=dict(color=ACCENT, size=16, line=dict(color=TEXT, width=2)),
            text=[f"Growth peak: {int(growth_peak_row['Discount (%)'])}%"],
            textposition="top center",
            textfont=dict(color=TEXT, size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_growth.update_layout(
            template="plotly_dark",
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
            title=dict(
                text="Sales growth by discount",
                font=dict(color=TEXT, size=15),
                x=0.02, xanchor="left",
            ),
            xaxis=dict(
                title="Discount (%)",
                tickmode="array", tickvals=ALL_DISCOUNTS, ticksuffix="%",
                gridcolor=BORDER, zerolinecolor=BORDER,
            ),
            yaxis=dict(
                title="Sales growth (ΔQ)",
                gridcolor=BORDER, zerolinecolor=BORDER,
                tickfont=dict(color=TEXT),
            ),
            margin=dict(l=40, r=20, t=60, b=40),
            height=340,
            showlegend=False,
        )

        st.plotly_chart(fig_growth, width="stretch")
        st.caption(
            f"Growth peaks at {int(growth_peak_row['Discount (%)'])}%. "
            f"This is the discount that maximizes sales volume under the frozen predictor."
        )

    # --------------------------------------------------
    # Chart B — Reward by discount
    # --------------------------------------------------
    with col_right:
        fig_reward = go.Figure()

        fig_reward.add_trace(go.Scatter(
            x=df["Discount (%)"], y=df["Reward"],
            mode="lines+markers",
            name="Reward (α=0.5, β=0.5)",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(color=ACCENT, size=7),
            hovertemplate="Discount %{x}%<br>Reward %{y:.4f}<extra></extra>",
        ))

        # Reward peak
        fig_reward.add_trace(go.Scatter(
            x=[int(reward_peak_row["Discount (%)"])],
            y=[float(reward_peak_row["Reward"])],
            mode="markers+text", name="Reward peak",
            marker=dict(color=ACCENT, size=16, line=dict(color=TEXT, width=2)),
            text=[f"Reward peak: {int(reward_peak_row['Discount (%)'])}%"],
            textposition="top center",
            textfont=dict(color=TEXT, size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

        # PPO choice (hollow marker)
        chosen_reward = float(
            df.loc[df["Discount (%)"] == recommended_discount, "Reward"].iloc[0]
        )
        fig_reward.add_trace(go.Scatter(
            x=[int(recommended_discount)],
            y=[chosen_reward],
            mode="markers+text", name="PPO choice",
            marker=dict(
                color="rgba(0,0,0,0)", size=16,
                symbol="circle-open",
                line=dict(color=TEXT, width=2.5),
            ),
            text=[f"PPO choice: {recommended_discount}%"],
            textposition="bottom center",
            textfont=dict(color=TEXT, size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_reward.update_layout(
            template="plotly_dark",
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
            title=dict(
                text="Reward by discount (α=0.5, β=0.5)",
                font=dict(color=TEXT, size=15),
                x=0.02, xanchor="left",
            ),
            xaxis=dict(
                title="Discount (%)",
                tickmode="array", tickvals=ALL_DISCOUNTS, ticksuffix="%",
                gridcolor=BORDER, zerolinecolor=BORDER,
            ),
            yaxis=dict(
                title="Reward",
                gridcolor=BORDER, zerolinecolor=BORDER,
                tickfont=dict(color=ACCENT),
            ),
            margin=dict(l=40, r=20, t=60, b=40),
            height=340,
            showlegend=False,
        )

        st.plotly_chart(fig_reward, width="stretch")
        st.caption(
            f"Reward peaks at {int(reward_peak_row['Discount (%)'])}%. "
            f"The PPO policy picks {recommended_discount}%. "
            f"The policy is frozen and was trained on a different reward signal, "
            f"so it does not exactly match the reward's argmax."
        )

    # --------------------------------------------------
    # One-line comparison summary
    # --------------------------------------------------
    if int(reward_peak_row["Discount (%)"]) == int(recommended_discount):
        st.success(
            f"PPO choice matches the reward peak at {int(recommended_discount)}% "
            f"under α=0.5, β=0.5."
        )
    else:
        chosen_reward = float(
            df.loc[df["Discount (%)"] == recommended_discount, "Reward"].iloc[0]
        )
        reward_gap = float(reward_peak_row["Reward"]) - chosen_reward
        st.info(
            f"Reward peaks at {int(reward_peak_row['Discount (%)'])}%, "
            f"but the frozen PPO policy picks {recommended_discount}%. "
            f"Under α=0.5, β=0.5, the reward at "
            f"{int(reward_peak_row['Discount (%)'])}% is "
            f"{reward_gap:+.4f} higher than at {recommended_discount}%."
        )

    st.markdown("### Prediction table")

    table = df[["Discount (%)", "Growth (ΔQ)", "Margin (ΔM)", "Reward"]].copy()
    table = table.sort_values("Reward", ascending=False).reset_index(drop=True)
    table["PPO picks"] = [
        "✓" if int(d) == recommended_discount else ""
        for d in table["Discount (%)"]
    ]
    for col in ["Growth (ΔQ)", "Margin (ΔM)", "Reward"]:
        table[col] = table[col].map(lambda v: f"{v:+.4f}")

    st.dataframe(table, hide_index=True, width="stretch")

    # --------------------------------------------------
    # LaTeX Equation Footnote
    # --------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        f"<span style='color:{MUTED}; font-size: 0.9em;'>Definitions (matching METHODS_AND_RESULTS.md §2):</span>", 
        unsafe_allow_html=True
    )
    
    col_eq1, col_eq2, col_eq3 = st.columns(3)
    with col_eq1:
        st.latex(r"\text{Growth } (\Delta Q) = \frac{Q_t - Q_{t-1}}{Q_{t-1}}")
    with col_eq2:
        st.latex(r"\text{Margin } (\Delta M) = \frac{M_t - M_{t-1}}{|M_{t-1}|}")
    with col_eq3:
        st.latex(r"\text{Reward } = \alpha \cdot \Delta M + \beta \cdot \Delta Q")

else:
    st.info("Choose a product and click **Generate recommendation** to run the trained models.")