import streamlit as st
import pandas as pd

from main import (
    load_data,
    prepare_data,
    clustering,
    classification
)


# ================= PAGE =================

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title(
    "🎓 Student Attendance & Performance Analysis"
)

st.write(
    "Student analysis using K-Means Clustering "
    "and Decision Tree Classification."
)


# ================= UPLOAD =================

file = st.file_uploader(
    "Upload Student CSV Dataset",
    type=["csv"]
)


if file:

    try:

        # ================= DATA =================

        df = load_data(file)

        df = prepare_data(df)

        st.success(
            "Dataset loaded successfully!"
        )


        # ================= PREVIEW =================

        st.subheader("📊 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # ================= INFORMATION =================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            df.shape[0]
        )

        c2.metric(
            "Columns",
            df.shape[1]
        )

        c3.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )


        # ================= PERFORMANCE =================

        st.subheader(
            "📝 Final Marks & Performance"
        )

        st.dataframe(
            df[
                ["final_marks", "Performance"]
            ].head(10),
            use_container_width=True
        )


        # =================================================
        # K-MEANS
        # =================================================

        st.header(
            "1️⃣ K-Means Clustering"
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        # Don't use ID or final marks
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

        if len(numeric_columns) >= 2:

            features = st.multiselect(
                "Select features for clustering",
                numeric_columns,
                default=numeric_columns[
                    :min(5, len(numeric_columns))
                ]
            )

            k = st.slider(
                "Number of Clusters",
                2,
                5,
                3
            )

            if st.button(
                "Run K-Means"
            ):

                try:

                    result, model = clustering(
                        df.copy(),
                        features,
                        k
                    )

                    st.success(
                        "Clustering completed!"
                    )

                    display_columns = [
                        col for col in [
                            "student_id",
                            "Student_ID",
                            "final_marks",
                            "Performance",
                            "Cluster"
                        ]
                        if col in result.columns
                    ]

                    st.dataframe(
                        result[display_columns],
                        use_container_width=True
                    )


                    st.subheader(
                        "📊 Cluster Summary"
                    )

                    summary = result.groupby(
                        "Cluster"
                    )[features].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"Clustering Error: {e}"
                    )

        else:

            st.warning(
                "This dataset does not have enough "
                "numeric columns for clustering."
            )


        # =================================================
        # DECISION TREE
        # =================================================

        st.header(
            "2️⃣ Decision Tree Classification"
        )

        st.write(
            "The model predicts the student's "
            "Performance category."
        )


        if st.button(
            "Run Classification"
        ):

            try:

                (
                    model,
                    accuracy,
                    report,
                    features
                ) = classification(df)


                # ================= ACCURACY =================

                st.subheader(
                    "🎯 Model Accuracy"
                )

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


                # ================= FEATURES =================

                st.subheader(
                    "🔢 Features Used"
                )

                st.write(
                    ", ".join(features)
                )


                # ================= REPORT =================

                st.subheader(
                    "📋 Classification Report"
                )

                report_df = pd.DataFrame(
                    report
                ).T

                if "support" in report_df.columns:

                    report_df["support"] = (
                        report_df["support"]
                        .astype(int)
                    )

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

                st.dataframe(
                    report_df,
                    use_container_width=True
                )


                # ================= CLASSES =================

                st.subheader(
                    "🎓 Performance Categories"
                )

                st.write(
                    "Fail • Average • Good • Excellent"
                )


            except Exception as e:

                st.error(
                    f"Classification Error: {e}"
                )


    except Exception as e:

        st.error(
            f"Dataset Error: {e}"
        )