import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go


@st.cache_resource
def load_model(model_path: str = "foodify_model.pkl"):
    """Load the offline-trained Foodify model safely."""
    try:
        obj = joblib.load(model_path)
    except FileNotFoundError:
        return None, None, None
    if isinstance(obj, dict) and "model" in obj:
        return (
            obj["model"],
            obj.get("feature_names"),
            obj.get("feature_importances"),
        )
    return obj, None, None


@st.cache_data
def load_history(csv_path: str = "foodify_diverse_data.csv") -> pd.DataFrame | None:
    """Load the synthetic diverse habit history."""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        return None


def project_future_weights(
    user_df: pd.DataFrame,
    start_week: int = 12,
    end_week: int = 20,
) -> pd.DataFrame:
    """
    Use average habits from the last 3 actual weeks (<= start_week)
    to project future weights from start_week to end_week.
    """
    actual = user_df[user_df["week_index"] <= start_week].copy()
    if actual.shape[0] < 3:
        return pd.DataFrame()

    tail3 = actual.tail(3)
    steps = tail3["avg_daily_steps"].mean()
    cardio = tail3["cardio_hours_weekly"].mean()
    fast_food = tail3["fast_food_meals_weekly"].mean()
    cals_per_meal = tail3["avg_calories_per_meal"].mean()

    # Start from actual weight at start_week
    start_row = actual[actual["week_index"] == start_week].iloc[0]
    current_weight = float(start_row["current_weight"])

    weeks = []
    weights = []
    weekly_intake = []

    for w in range(start_week, end_week + 1):
        noise_cals_per_meal = np.random.normal(0.0, 80.0)
        noisy_cals = cals_per_meal + noise_cals_per_meal
        intake = noisy_cals * 21 + fast_food * 500

        base_delta_kg = (
            steps * 0.05
            + cardio * 200.0
            - fast_food * 500.0
            - (noisy_cals * 3.0)
        ) / 7700.0
        noise_delta = np.random.normal(0.0, 0.05)
        delta_kg = base_delta_kg + noise_delta

        current_weight = current_weight + delta_kg

        weeks.append(w)
        weights.append(current_weight)
        weekly_intake.append(intake)

    return pd.DataFrame(
        {
            "week_index": weeks,
            "predicted_weight": weights,
            "predicted_weekly_intake": weekly_intake,
        }
    )


def build_feature_frame_for_model(
    steps: float,
    cardio_hours: float,
    fast_food_meals: float,
    calories_per_meal: float,
    current_weight: float,
    feature_names: list[str] | None,
) -> pd.DataFrame:
    """Assemble the feature vector for the classification model."""
    data = {
        "avg_daily_steps": [steps],
        "cardio_hours_weekly": [cardio_hours],
        "fast_food_meals_weekly": [fast_food_meals],
        "avg_calories_per_meal": [calories_per_meal],
        "current_weight": [current_weight],
    }
    df = pd.DataFrame(data)
    if feature_names is not None:
        df = df[feature_names]
    return df


def main() -> None:
    st.set_page_config(
        page_title="Foodify – Trajectory Coach",
        page_icon="🍽️",
        layout="wide",
    )

    st.title("Foodify Trajectory Dashboard")
    st.subheader("High-variance habits with an AI-powered weight trajectory")

    history = load_history("foodify_diverse_data.csv")
    model, feature_names, feature_importances = load_model("foodify_model.pkl")

    if history is None:
        st.error(
            "Could not load `foodify_diverse_data.csv`. "
            "Run `generate_data.py` from the project root first."
        )
        return

    if model is None:
        st.error(
            "Could not load `foodify_model.pkl`. "
            "Run `train_model.py` after generating the data."
        )
        return

    # --- Sidebar: user selection ---------------------------------------------------
    st.sidebar.title("User Selection")
    available_users = sorted(history["username"].unique())
    default_user = available_users[0] if available_users else ""
    username = st.sidebar.text_input(
        "Username",
        value=default_user,
        help="Use one of the simulated users, e.g. `user_1`.",
    )

    user_df = history[history["username"] == username].copy()
    if user_df.empty:
        st.warning(
            f"No history found for `{username}`. "
            f"Try one of: {', '.join(available_users[:5])}..."
        )
        return

    user_df = user_df.sort_values("week_index")
    actual_df = user_df[user_df["week_index"] <= 12].copy()
    if actual_df.empty:
        st.warning("No actual history for weeks 1–12 for this user.")
        return

    forecast_df = project_future_weights(user_df, start_week=12, end_week=20)

    initial_weight = float(user_df["initial_weight"].iloc[0])
    target_weight = float(user_df["target_weight"].iloc[0])
    current_weight = float(actual_df["current_weight"].iloc[-1])

    # --- Trajectory chart with Plotly ---------------------------------------------
    st.markdown(f"### Weight trajectory for `{username}`")

    fig = go.Figure()

    # Actual weights (weeks 1–12)
    fig.add_trace(
        go.Scatter(
            x=actual_df["week_index"],
            y=actual_df["current_weight"],
            mode="lines+markers",
            name="Actual weight",
            line=dict(color="#1f77b4"),
            marker=dict(size=7),
            customdata=actual_df[["weekly_intake"]].values,
            hovertemplate=(
                "Week %{x}<br>Actual weight: %{y:.1f} kg"
                "<br>Weekly intake: %{customdata[0]:.0f} kcal<extra></extra>"
            ),
        )
    )

    # Predicted weights (weeks 12–20)
    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["week_index"],
                y=forecast_df["predicted_weight"],
                mode="lines+markers",
                name="Predicted weight",
                line=dict(color="#ff7f0e", dash="dash"),
                marker=dict(size=6),
                customdata=forecast_df[["predicted_weekly_intake"]].values,
                hovertemplate=(
                    "Week %{x}<br>Predicted weight: %{y:.1f} kg"
                    "<br>Predicted weekly intake: %{customdata[0]:.0f} kcal"
                    "<extra></extra>"
                ),
            )
        )

    # Target line (-7.5% weight)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 21)),
            y=[target_weight] * 20,
            mode="lines",
            name="Target weight (-7.5%)",
            line=dict(color="#2ca02c", dash="dot"),
            hovertemplate="Target weight: %{y:.1f} kg<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Weight (kg)",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics and ETA ----------------------------------------------------------
    total_lost = initial_weight - current_weight
    avg_weekly_deficit = float(actual_df["weekly_deficit"].mean())

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Initial weight (kg)", f"{initial_weight:.1f}")
    with m2:
        st.metric("Current weight (kg)", f"{current_weight:.1f}")
    with m3:
        st.metric(
            "Total weight lost",
            f"{total_lost:.1f} kg",
            delta=f"{(total_lost/initial_weight)*100:.1f}% of body weight",
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Target weight (kg)", f"{target_weight:.1f}")
    with c2:
        st.metric("Average weekly deficit (kcal)", f"{avg_weekly_deficit:.0f}")

    # ETA to goal from trajectory
    eta_text = "Goal not reached by Week 20."
    eta_week = None

    # Combine actual (1–12) and predicted (13–20) series
    combined_weeks = list(actual_df["week_index"])
    combined_weights = list(actual_df["current_weight"])
    if not forecast_df.empty:
        for w, wt in zip(
            forecast_df["week_index"].tolist(),
            forecast_df["predicted_weight"].tolist(),
        ):
            if w > 12:  # avoid duplicating week 12
                combined_weeks.append(w)
                combined_weights.append(wt)

    for w, wt in zip(combined_weeks, combined_weights):
        if wt <= target_weight:
            eta_week = w
            break

    if eta_week is not None:
        if eta_week <= 12:
            eta_text = f"Target expected to be reached around **Week {eta_week} (already in actual data)**."
        else:
            eta_text = f"Target expected to be reached around **Week {eta_week} (forecast)**."

    st.markdown(f"### ETA to goal\n{eta_text}")

    # --- Habit insights from model importances ------------------------------------
    st.markdown("### Habit insights from the model")

    if feature_importances is not None and feature_names is not None:
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importances}
        ).sort_values("importance", ascending=False)

        helpful_candidates = [
            "avg_daily_steps",
            "cardio_hours_weekly",
        ]
        harming_candidates = [
            "fast_food_meals_weekly",
            "avg_calories_per_meal",
            "current_weight",
        ]

        helpful = [
            f
            for f in fi_df["feature"].tolist()
            if f in helpful_candidates
        ][:2]
        harming = [
            f
            for f in fi_df["feature"].tolist()
            if f in harming_candidates
        ][:1]

        st.markdown(
            "- **Top 2 habits helping you**: "
            + (", ".join(f"`{h}``" for h in helpful) if helpful else "`avg_daily_steps`, `cardio_hours_weekly`")
        )
        st.markdown(
            "- **Top habit hurting you**: "
            + (f"`{harming[0]}`" if harming else "`fast_food_meals_weekly`")
        )

        st.markdown("Model feature importance (global view):")
        st.bar_chart(fi_df.set_index("feature"))

    # Optional: AI probability at current habits (last 3 weeks)
    tail3 = actual_df.tail(3)
    steps = tail3["avg_daily_steps"].mean()
    cardio = tail3["cardio_hours_weekly"].mean()
    fast_food = tail3["fast_food_meals_weekly"].mean()
    cals_per_meal = tail3["avg_calories_per_meal"].mean()

    X_for_model = build_feature_frame_for_model(
        steps=steps,
        cardio_hours=cardio,
        fast_food_meals=fast_food,
        calories_per_meal=cals_per_meal,
        current_weight=current_weight,
        feature_names=feature_names,
    )
    proba = model.predict_proba(X_for_model)[0]
    prob_success = float(proba[1])
    st.markdown(
        f"**AI estimate of hitting -7.5% goal at your recent pace**: "
        f"`{prob_success*100:.1f}%`"
    )


if __name__ == "__main__":
    main()




