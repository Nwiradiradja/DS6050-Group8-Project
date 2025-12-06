import os
import glob
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline

def _download_us_accidents_csv():
    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print("Downloaded to:", path)

    csv_candidates = glob.glob(os.path.join(path, "**", "US_Accidents*.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    if not csv_candidates:
        raise FileNotFoundError("Couldn't find a CSV in the downloaded dataset.")

    csv_sizes = [(p, os.path.getsize(p)) for p in csv_candidates]
    csv_path = sorted(csv_sizes, key=lambda x: x[1], reverse=True)[0][0]
    print("Using CSV:", csv_path)
    return csv_path

def _simplify_weather(w):
    w = str(w).lower()
    if "thunder" in w:
        return "Thunderstorm"
    if "snow" in w or "sleet" in w or "blizzard" in w:
        return "Snow"
    if "rain" in w or "drizzle" in w or "shower" in w:
        return "Rain"
    if "fog" in w or "mist" in w or "haze" in w or "smoke" in w:
        return "Fog"
    if "cloud" in w or "overcast" in w:
        return "Cloudy"
    if "clear" in w or "fair" in w:
        return "Clear"
    return "Other"

def load_and_prepare_data(sample_n: int = 300_000, random_state: int = 42):

    csv_path = _download_us_accidents_csv()

    use_cols = [
        "Severity", "Start_Time", "Weather_Condition",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
        "Precipitation(in)", "Sunrise_Sunset",
    ]

    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    present_cols = [c for c in use_cols if c in header_cols]

    # Sampling
    if sample_n is None:
        df = pd.read_csv(csv_path, usecols=present_cols, low_memory=False)
    else:
        total_rows = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
        sample_n = min(sample_n, total_rows)
        keep_idx = set(np.random.choice(total_rows, size=sample_n, replace=False))

        def row_use(i):
            return (i - 1) in keep_idx

        df = pd.read_csv(
            csv_path,
            usecols=present_cols,
            low_memory=False,
            skiprows=lambda i: i > 0 and not row_use(i),
        )

    print("Sampled shape:", df.shape)

    # TARGET
    df = df[df["Severity"].isin([1, 2, 3, 4])].copy()
    df["target"] = df["Severity"].apply(lambda x: 0 if x in [1, 2] else 1)

    # TIME FEATURES
    ts = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9)) |
        ((df["hour"] >= 16) & (df["hour"] <= 18))
    ).astype(int)

    # WEATHER FEATURES
    if "Weather_Condition" in df.columns:
        df["Weather_Simple"] = df["Weather_Condition"].apply(_simplify_weather)
    else:
        df["Weather_Simple"] = "Other"

    df["Daylight"] = df["Sunrise_Sunset"].map({"Day": 1, "Night": 0}) if "Sunrise_Sunset" in df.columns else np.nan

    if "Visibility(mi)" in df.columns:
        df["low_visibility"] = (df["Visibility(mi)"] < 3.0).astype(int)

    keep_cols = [
        "target",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)",
        "hour", "is_weekend", "is_rush_hour", "low_visibility",
        "Weather_Simple", "Daylight",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df_full = df.copy()

    # SPLIT X/Y
    y = df["target"].values
    X = df.drop(columns=["target"])

    num_candidates = [
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)",
        "hour", "is_weekend", "is_rush_hour", "low_visibility",
    ]
    cat_candidates = ["Weather_Simple", "Daylight"]

    num_cols = [c for c in num_candidates if c in X.columns]
    cat_cols = [c for c in cat_candidates if c in X.columns]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    classes = np.unique(y_train)
    class_weights_train = {
        int(c): float(w)
        for c, w in zip(
            classes,
            compute_class_weight("balanced", classes=classes, y=y_train)
        )
    }

    print("Class weights (train):", class_weights_train)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocess,
        df_full,
        class_weights_train,
    )
