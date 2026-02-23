import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def load_data(csv_path: str = "foodify_diverse_data.csv") -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Could not find `{csv_path}`. "
            "Run `generate_data.py` first to create the dataset."
        )
        return None


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare features and labels for the Foodify model.

    Features:
        - avg_daily_steps
        - cardio_hours_weekly
        - fast_food_meals_weekly
        - avg_calories_per_meal
        - current_weight
    Target:
        - will_hit_target (Success/Failure -> 1/0)
    """
    df_encoded = df.copy()

    target_map = {"Failure": 0, "Success": 1}
    df_encoded["will_hit_target_encoded"] = df_encoded["will_hit_target"].map(
        target_map
    )

    feature_cols = [
        "avg_daily_steps",
        "cardio_hours_weekly",
        "fast_food_meals_weekly",
        "avg_calories_per_meal",
        "current_weight",
    ]

    X = df_encoded[feature_cols]
    y = df_encoded["will_hit_target_encoded"]
    return X, y, feature_cols


def train_and_save_model(
    csv_path: str = "foodify_diverse_data.csv",
    model_path: str = "foodify_model.pkl",
) -> None:
    df = load_data(csv_path)
    if df is None:
        return

    X, y, feature_cols = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(
        classification_report(
            y_test, y_pred, target_names=["Failure", "Success"]
        )
    )

    importances = clf.feature_importances_
    for name, imp in sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"Feature importance - {name}: {imp:.3f}")

    bundle = {
        "model": clf,
        "feature_names": feature_cols,
        "feature_importances": importances,
    }
    joblib.dump(bundle, model_path)
    print(f"Saved Foodify model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()




