import numpy as np
import pandas as pd

from mirror.nodes import GaussianNode, CategoricalNode


def generate_foodify_diverse_data(
    n_users: int = 50,
    n_weeks: int = 20,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Generate a high-variance, persona-driven dataset for Foodify.

    - 50 users over 20 labelled weeks ('Week 1', ..., 'Week 20')
    - Persona types control baselines for weight and habits
    - Weekly weight change includes random noise for realism
    - Binary target: `will_hit_target` (7.5% loss by week 20)
    """
    if random_state is not None:
        np.random.seed(random_state)

    usernames = [f"user_{i}" for i in range(1, n_users + 1)]
    total_rows = n_users * n_weeks

    # Persona assignment per user
    persona_types = ["Sedentary", "Active", "Inconsistent"]
    persona_probs = [0.4, 0.3, 0.3]
    persona_node = CategoricalNode(
        name="persona_type",
        parameters={p: prob for p, prob in zip(persona_types, persona_probs)},
        sample_n=n_users,
    )
    persona_per_user = persona_node.instantiate_values()

    # Map personas to baseline parameters
    persona_weight_mu = {
        "Sedentary": 100.0,
        "Active": 78.0,
        "Inconsistent": 90.0,
    }
    persona_weight_sigma = {
        "Sedentary": 8.0,
        "Active": 6.0,
        "Inconsistent": 7.0,
    }
    persona_steps_mu = {
        "Sedentary": 4000.0,
        "Active": 11000.0,
        "Inconsistent": 7000.0,
    }
    persona_cardio_mu = {
        "Sedentary": 0.5,
        "Active": 4.0,
        "Inconsistent": 2.0,
    }
    persona_fast_food_mu = {
        "Sedentary": 5.0,
        "Active": 1.0,
        "Inconsistent": 3.0,
    }
    persona_cals_mu = {
        "Sedentary": 800.0,
        "Active": 650.0,
        "Inconsistent": 725.0,
    }

    # Initial weight per user using persona-specific Gaussians
    initial_weight_per_user = np.zeros(n_users)
    for i, persona in enumerate(persona_per_user):
        mu = persona_weight_mu[persona]
        sigma = persona_weight_sigma[persona]
        initial_weight_per_user[i] = np.random.normal(mu, sigma)
    initial_weight_per_user = np.clip(initial_weight_per_user, 70.0, 120.0)

    # Username, week labels, persona per row
    username_sequence = np.repeat(usernames, n_weeks)
    week_indices = np.tile(np.arange(1, n_weeks + 1), n_users)
    week_labels = np.array([f"Week {i}" for i in week_indices])
    persona_sequence = np.repeat(persona_per_user, n_weeks)

    user_to_weight = dict(zip(usernames, initial_weight_per_user))
    initial_weight = np.array([user_to_weight[u] for u in username_sequence])

    target_weight = initial_weight * 0.925

    # Use GaussianNode for global habit variability, then adjust by persona baselines
    steps_node = GaussianNode(
        name="avg_daily_steps",
        sample_n=total_rows,
        miu=7000.0,
        var=3500.0**2,
    )
    base_steps = steps_node.instantiate_values()

    cardio_node = GaussianNode(
        name="cardio_hours_weekly",
        sample_n=total_rows,
        miu=2.0,
        var=2.5**2,
    )
    base_cardio = cardio_node.instantiate_values()

    calories_node = GaussianNode(
        name="avg_calories_per_meal",
        sample_n=total_rows,
        miu=725.0,
        var=175.0**2,
    )
    base_cals = calories_node.instantiate_values()

    # Persona adjustments and clipping
    avg_daily_steps = np.zeros(total_rows)
    cardio_hours_weekly = np.zeros(total_rows)
    fast_food_meals_weekly = np.zeros(total_rows)
    avg_calories_per_meal = np.zeros(total_rows)

    for i in range(total_rows):
        persona = persona_sequence[i]
        avg_daily_steps[i] = base_steps[i] + (persona_steps_mu[persona] - 7000.0)
        cardio_hours_weekly[i] = base_cardio[i] + (persona_cardio_mu[persona] - 2.0)
        # Poisson-like fast food with persona mean
        lam = max(0.5, persona_fast_food_mu[persona])
        fast_food_meals_weekly[i] = np.random.poisson(lam=lam)
        avg_calories_per_meal[i] = base_cals[i] + (persona_cals_mu[persona] - 725.0)

    avg_daily_steps = np.clip(avg_daily_steps, 1000, 25000)
    cardio_hours_weekly = np.clip(cardio_hours_weekly, 0.0, 14.0)
    fast_food_meals_weekly = np.clip(fast_food_meals_weekly, 0, 21)
    avg_calories_per_meal = np.clip(avg_calories_per_meal, 400, 1600)

    df = pd.DataFrame(
        {
            "username": username_sequence,
            "week_index": week_indices,
            "week_label": week_labels,
            "persona_type": persona_sequence,
            "initial_weight": initial_weight,
            "target_weight": target_weight,
            "avg_daily_steps": avg_daily_steps,
            "cardio_hours_weekly": cardio_hours_weekly,
            "fast_food_meals_weekly": fast_food_meals_weekly,
            "avg_calories_per_meal": avg_calories_per_meal,
        }
    )

    # Weight dynamics per user over time with high-variance noise
    df = df.sort_values(["username", "week_index"]).reset_index(drop=True)

    current_weight = df["initial_weight"].copy()
    weekly_intake = np.zeros(total_rows)
    weekly_burn = np.zeros(total_rows)
    weekly_deficit = np.zeros(total_rows)
    weekly_weight_change = np.zeros(total_rows)

    for i in range(total_rows):
        steps = df.loc[i, "avg_daily_steps"]
        cardio = df.loc[i, "cardio_hours_weekly"]
        fast_food = df.loc[i, "fast_food_meals_weekly"]
        cals_per_meal = df.loc[i, "avg_calories_per_meal"]

        # Random holiday/stress noise on calories (per-week, in meal-equivalent units)
        noise_cals_per_meal = np.random.normal(0.0, 120.0)
        noisy_cals_per_meal = cals_per_meal + noise_cals_per_meal

        weekly_intake_val = noisy_cals_per_meal * 21 + fast_food * 500
        weekly_burn_val = (
            steps * 0.05 + cardio * 200.0 - fast_food * 500.0
        )

        # High-variance weekly weight change with additional noise
        base_delta_kg = (
            steps * 0.05
            + cardio * 200.0
            - fast_food * 500.0
            - (noisy_cals_per_meal * 3.0)
        ) / 7700.0
        noise_delta = np.random.normal(0.0, 0.08)
        delta_kg = base_delta_kg + noise_delta

        weekly_intake[i] = weekly_intake_val
        weekly_burn[i] = weekly_burn_val
        weekly_deficit[i] = weekly_burn_val - weekly_intake_val
        weekly_weight_change[i] = delta_kg

    # Apply weight changes sequentially per user
    current_weight = np.zeros(total_rows)
    for user in usernames:
        mask = df["username"] == user
        idx = np.where(mask)[0]
        idx_sorted = idx[np.argsort(df.loc[idx, "week_index"].values)]
        w = df.loc[idx_sorted[0], "initial_weight"]
        for j in idx_sorted:
            w = w + weekly_weight_change[j]
            current_weight[j] = w

    df["current_weight"] = current_weight
    df["weekly_intake"] = weekly_intake
    df["weekly_burn"] = weekly_burn
    df["weekly_deficit"] = weekly_deficit

    # Target success flag based on week 20 outcome (7.5% loss)
    final_success = {}
    for u in usernames:
        user_mask = df["username"] == u
        user_rows = df[user_mask]
        week20_row = user_rows[user_rows["week_index"] == n_weeks].iloc[0]
        hit = week20_row["current_weight"] <= week20_row["initial_weight"] * 0.925
        final_success[u] = "Success" if hit else "Failure"

    df["will_hit_target"] = df["username"].map(final_success)

    return df


def main() -> None:
    df = generate_foodify_diverse_data()
    df.to_csv("foodify_diverse_data.csv", index=False)


if __name__ == "__main__":
    main()



