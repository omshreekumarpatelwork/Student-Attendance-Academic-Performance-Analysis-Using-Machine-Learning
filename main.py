import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ================= LOAD CSV =================

def load_data(file):
    if not file.name.endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    return pd.read_csv(file)


# ================= PREPROCESSING =================

def prepare_data(df):

    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")

    # Remove completely empty assignment column
    if "assignments_submitted" in df.columns:
        if df["assignments_submitted"].isna().all():
            df = df.drop(columns=["assignments_submitted"])

    # Add final marks if dataset does not have them
    if "final_marks" not in df.columns:

        np.random.seed(42)

        df["final_marks"] = np.random.randint(
            40, 101, len(df)
        )

    # Create performance category
    df["Performance"] = pd.cut(
        df["final_marks"],
        bins=[0, 40, 60, 75, 100],
        labels=[
            "Fail",
            "Average",
            "Good",
            "Excellent"
        ],
        include_lowest=True
    )

    return df


# ================= K-MEANS =================

def clustering(df, features, k):

    if len(features) < 2:
        raise ValueError(
            "Please select at least 2 features."
        )

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.dropna(axis=1, how="all")

    if X.shape[1] < 2:
        raise ValueError(
            "At least 2 numeric features are required."
        )

    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    return df, model


# ================= DECISION TREE =================

def classification(df):

    # Use all useful numeric columns automatically
    features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Remove columns that should NOT be inputs
    exclude = [
        "final_marks",
        "student_id",
        "Student_ID",
        "id",
        "ID"
    ]

    features = [
        f for f in features
        if f not in exclude
    ]

    if len(features) < 2:
        raise ValueError(
            "Not enough numeric features for classification."
        )

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())

    y = df["Performance"].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "Performance contains only one class."
        )

    # Try stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0
    )

    return (
        model,
        accuracy,
        report,
        features
    )