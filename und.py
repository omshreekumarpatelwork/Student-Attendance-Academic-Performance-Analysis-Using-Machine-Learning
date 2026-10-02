# =========================
# IMPORTS
# =========================

import streamlit as st
# Streamlit → website banane ke liye

import pandas as pd
# Pandas → CSV data handle karne ke liye

import numpy as np
# NumPy → numerical operations ke liye

from sklearn.preprocessing import StandardScaler
# Data ko standard scale par convert karta hai

from sklearn.cluster import KMeans
# Students ko groups/clusters mein divide karta hai

from sklearn.model_selection import train_test_split
# Dataset ko training aur testing data mein divide karta hai

from sklearn.tree import DecisionTreeClassifier
# Decision Tree classification model

from sklearn.metrics import accuracy_score, classification_report
# Model ki accuracy aur classification report calculate karta hai


# =========================
# DATA PREPARATION
# =========================

def prepare_data(df):
    # Dataset ko clean aur prepare karta hai

    df = df.drop_duplicates()
    # Duplicate rows remove karta hai

    df = df.dropna(axis=1, how="all")
    # Completely empty columns remove karta hai

    if "final_marks" not in df.columns:
        # Check karta hai ki final_marks column hai ya nahi

        np.random.seed(42)
        # Same random values reproduce karne ke liye

        df["final_marks"] = np.random.randint(
            40, 101, len(df)
        )
        # 40 se 100 ke beech marks generate karta hai

    df["Performance"] = pd.cut(
        df["final_marks"],
        bins=[0, 40, 60, 75, 100],
        labels=["Fail", "Average", "Good", "Excellent"],
        include_lowest=True
    )
    # Final marks ko performance categories mein convert karta hai

    return df
    # Prepared dataset return karta hai


# =========================
# K-MEANS CLUSTERING
# =========================

def clustering(df, features, k):
    # K-Means clustering perform karta hai

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )
    # Selected columns ko numeric format mein convert karta hai

    X = X.fillna(X.median())
    # Missing values ko median se replace karta hai

    scaler = StandardScaler()
    # Scaling object create karta hai

    X_scaled = scaler.fit_transform(X)
    # Data ko standardize karta hai

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    # K-Means model create karta hai

    df["Cluster"] = model.fit_predict(X_scaled)
    # Har student ko ek cluster assign karta hai

    return df, model
    # Clustered data aur model return karta hai


# =========================
# DECISION TREE
# =========================

def classification(df):
    # Student performance classification karta hai

    features = df.select_dtypes(
        include="number"
    ).columns.tolist()
    # Dataset ke numerical columns select karta hai

    exclude = [
        "final_marks",
        "student_id",
        "Student_ID",
        "id",
        "ID"
    ]
    # In columns ko model se exclude karta hai

    features = [
        f for f in features
        if f not in exclude
    ]
    # Useful features ki list banata hai

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )
    # Features ko numeric format mein convert karta hai

    X = X.fillna(X.median())
    # Missing values ko median se fill karta hai

    y = df["Performance"].astype(str)
    # Performance target variable hai

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    # 80% training aur 20% testing data banata hai

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    # Decision Tree model create karta hai

    model.fit(X_train, y_train)
    # Model ko training data se train karta hai

    prediction = model.predict(X_test)
    # Test data ki prediction karta hai

    accuracy = accuracy_score(
        y_test,
        prediction
    )
    # Model accuracy calculate karta hai

    report = classification_report(
        y_test,
        prediction,
        output_dict=True,
        zero_division=0
    )
    # Precision, recall, F1-score aur support calculate karta hai

    return model, accuracy, report, features
    # Results return karta hai


# =========================
# STREAMLIT APP
# =========================

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)
# Website ka page configuration


st.title(
    "🎓 Student Attendance & Performance Analysis"
)
# Main website heading


st.write(
    "Student analysis using K-Means Clustering "
    "and Decision Tree Classification."
)
# Project ka short description


file = st.file_uploader(
    "Upload Student CSV Dataset",
    type=["csv"]
)
# Sirf CSV file upload karne deta hai


if file:
    # Agar user ne CSV upload ki hai

    try:

        df = pd.read_csv(file)
        # CSV ko DataFrame mein read karta hai

        df = prepare_data(df)
        # Dataset ko clean aur prepare karta hai

        st.success(
            "Dataset loaded successfully!"
        )
        # Successful upload message


        st.subheader("📊 Dataset Preview")
        # Preview heading

        st.dataframe(
            df.head(10),
            use_container_width=True
        )
        # First 10 rows show karta hai


        col1, col2, col3 = st.columns(3)
        # 3 display columns create karta hai

        col1.metric("Rows", df.shape[0])
        # Total rows show karta hai

        col2.metric("Columns", df.shape[1])
        # Total columns show karta hai

        col3.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )
        # Total missing values show karta hai


        # =========================
        # K-MEANS SECTION
        # =========================

        st.header("1️⃣ K-Means Clustering")
        # K-Means heading

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()
        # Numerical columns identify karta hai

        numeric_columns = [
            col for col in numeric_columns
            if col not in [
                "final_marks",
                "student_id",
                "Student_ID",
                "id",
                "ID"
            ]
        ]
        # ID aur final_marks ko remove karta hai

        if len(numeric_columns) >= 2:

            features = st.multiselect(
                "Select features for clustering",
                numeric_columns,
                default=numeric_columns[
                    :min(5, len(numeric_columns))
                ]
            )
            # User clustering ke features select karta hai

            k = st.slider(
                "Number of Clusters",
                2,
                5,
                3
            )
            # Number of clusters select karta hai

            if st.button("Run K-Means"):
                # Button click hone par clustering run hoti hai

                try:

                    result, model = clustering(
                        df.copy(),
                        features,
                        k
                    )
                    # K-Means function call karta hai

                    st.success(
                        "Clustering completed!"
                    )
                    # Success message

                    st.dataframe(
                        result,
                        use_container_width=True
                    )
                    # Clustered dataset show karta hai

                    st.subheader(
                        "📊 Cluster Summary"
                    )
                    # Summary heading

                    summary = result.groupby(
                        "Cluster"
                    )[features].mean().round(2)
                    # Har cluster ka average calculate karta hai

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )
                    # Cluster summary display karta hai

                except Exception as e:

                    st.error(
                        f"Clustering Error: {e}"
                    )
                    # Clustering error show karta hai


        # =========================
        # CLASSIFICATION SECTION
        # =========================

        st.header(
            "2️⃣ Decision Tree Classification"
        )
        # Classification heading

        st.write(
            "The model predicts the student's Performance category."
        )
        # Model ka purpose


        if st.button("Run Classification"):
            # Classification button click hone par

            try:

                model, accuracy, report, features = classification(df)
                # Decision Tree function call karta hai

                st.subheader(
                    "🎯 Model Accuracy"
                )
                # Accuracy heading

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )
                # Accuracy percentage show karta hai


                st.subheader(
                    "🔢 Features Used"
                )
                # Features heading

                st.write(
                    ", ".join(features)
                )
                # Model mein used features show karta hai


                st.subheader(
                    "📋 Classification Report"
                )
                # Classification report heading

                report_df = pd.DataFrame(
                    report
                ).T
                # Report ko table mein convert karta hai


                if "support" in report_df.columns:

                    report_df["support"] = (
                        report_df["support"]
                        .astype(int)
                    )
                    # Support values ko integer mein convert karta hai


                for column in [
                    "precision",
                    "recall",
                    "f1-score"
                ]:

                    if column in report_df.columns:

                        report_df[column] = (
                            report_df[column]
                            .round(2)
                        )
                        # Scores ko 2 decimal places tak round karta hai


                st.dataframe(
                    report_df,
                    use_container_width=True
                )
                # Clean classification report table show karta hai


                st.subheader(
                    "🎓 Performance Categories"
                )
                # Performance category heading

                st.write(
                    "Fail • Average • Good • Excellent"
                )
                # Categories display karta hai


            except Exception as e:

                st.error(
                    f"Classification Error: {e}"
                )
                # Classification error show karta hai


    except Exception as e:

        st.error(
            f"Dataset Error: {e}"
        )
        # Dataset related error show karta hai


else:

    st.info(
        "Please upload a CSV dataset to begin."
    )
    # File upload nahi hone par message