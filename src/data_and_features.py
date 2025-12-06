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


def _download_us_accidents_csv():
    """
    Download the US Accidents dataset via kagglehub and return the path
    to the largest CSV file (e.g., US_Accidents_March23.csv).
    """
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
    """
    Map raw Weather_Condition strings into a small set of categories.
    """
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
    """
    Download the US Accidents dataset, sample rows, engineer features,
    build train/val/test splits, and return everything needed by the
    baseline and ablation scripts.

    Returns
    -------
    X_train, X_val, X_test : pandas.DataFrame
        Feature matrices for each split.
    y_train, y_val, y_test : numpy.ndarray
        Binary target arrays for each split (0=minor, 1=severe).
    preprocess : sklearn.compose.ColumnTransformer
        Fitted ColumnTransformer that handles numeric + categorical features.
    df_full : pandas.DataFrame
        Engineered dataframe (after keep_cols) with target + all features,
        used for ablation 3 (minus-one feature groups).
    class_weights_train : dict
        Mapping {0: weight_for_minor, 1: weight_for_severe} computed on train only.
    """
    csv_path = _download_us_accidents_csv()

    # -----------------------------
    # 1. Read sample of the CSV
    # -----------------------------
    use_cols = [
        "Severity", "Start_Time", "Weather_Condition",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
        "Precipitation(in)", "Sunrise_Sunset",
    ]

    # Only keep columns actually present in this version
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    present_cols = [c for c in use_cols if c in header_cols]

    if sample_n is None:
        df = pd.read_csv(csv_path, usecols=present_cols, low_memory=False)
    else:
        # Sample rows by index without reading the entire dataset into memory
        total_rows = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
        sample_n = min(sample_n, total_rows)
        keep_idx = set(np.random.choice(total_rows, size=sample_n, replace=False))

        def row_use(i):
            # i=0 is header, so we offset by 1
            return (i - 1) in keep_idx

        df = pd.read_csv(
            csv_path,
            usecols=present_cols,
            low_memory=False,
            skiprows=lambda i: i > 0 and not row_use(i),
        )

    print("Sampled shape:", df.shape)

    # -----------------------------
    # 2. Target + feature engineering
    # -----------------------------
    # Keep Severity 1–4 and build binary target: 0=minor (1,2), 1=severe (3,4)
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

    if "Sunrise_Sunset" in df.columns:
        df["Daylight"] = df["Sunrise_Sunset"].map({"Day": 1, "Night": 0})
    else:
        df["Daylight"] = np.nan

    if "Visibility(mi)" in df.columns:
        df["low_visibility"] = (df["Visibility(mi)"] < 3.0).astype(int)

    # Final columns we care about
    keep_cols = [
        "target",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)",
        "hour", "is_weekend", "is_rush_hour", "low_visibility",
        "Weather_Simple", "Daylight",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Save a copy for ablations (minus-one feature groups)
    df_full = df.copy()

    # -----------------------------
    # 3. Split X / y and build preprocess
    # -----------------------------
    y = df["target"].values
    X = df.drop(columns=["target"])

    # Base candidates for numeric and categorical columns
    num_candidates = [
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)",
        "hour", "is_weekend", "is_rush_hour", "low_visibility",
    ]
    cat_candidates = ["Weather_Simple", "Daylight"]

    num_cols = [c for c in num_candidates if c in X.columns]
    cat_cols = [c for c in cat_candidates if c in X.columns]

    num_pipe = SimpleImputer(strategy="median")
    num_pipe = PipelineOrSimple(num_pipe, scaler=True)

    # but we need imports for Pipeline; simpler: build full pipeline explicitly
    from sklearn.pipeline import Pipeline

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

    # Train / val / test split (80 / 10 / 10)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    print("Split shapes:", X_train.shape, X_val.shape, X_test.shape)

    # -----------------------------
    # 4. Class weights from train only
    # -----------------------------
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights_train = {int(c): float(w) for c, w in zip(classes, class_weights_array)}
    print("Class weights (train):", class_weights_train)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocess,
        df_full,
        class_weights_train,
    )