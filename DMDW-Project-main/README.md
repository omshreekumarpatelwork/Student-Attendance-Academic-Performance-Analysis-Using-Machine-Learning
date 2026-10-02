# 🎓 Student Attendance & Performance Analysis

An interactive **Machine Learning web application built with Streamlit** for analyzing student attendance, academic performance, and student behavior using **K-Means Clustering** and **Decision Tree Classification**.

The application allows the user to upload a **CSV dataset**, preprocess the data, analyze student groups using clustering, and classify students into different performance categories.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Motivation](#-motivation)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Machine Learning Techniques](#-machine-learning-techniques)
- [Dataset](#-dataset)
- [Dataset Attributes](#-dataset-attributes)
- [Data Preprocessing](#-data-preprocessing)
- [Final Marks](#-final-marks)
- [Performance Categories](#-performance-categories)
- [K-Means Clustering](#-k-means-clustering)
- [Decision Tree Classification](#-decision-tree-classification)
- [Model Evaluation](#-model-evaluation)
- [Project Workflow](#-project-workflow)
- [System Architecture](#-system-architecture)
- [Application Interface](#-application-interface)
- [Technologies Used](#-technologies-used)
- [Python Libraries](#-python-libraries)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [How to Use](#-how-to-use)
- [Output](#-output)
- [Data Mining Concepts Used](#-data-mining-concepts-used)
- [Use Cases](#-use-cases)
- [Advantages](#-advantages)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Testing](#-testing)
- [Error Handling](#-error-handling)
- [Academic Relevance](#-academic-relevance)
- [Team](#-team)
- [Conclusion](#-conclusion)
- [License](#-license)

---

# 📖 Project Overview

**Student Attendance & Performance Analysis** is a web-based Machine Learning application developed using **Python and Streamlit**.

The main purpose of this project is to analyze student-related data and identify patterns in student performance.

The application uses two Machine Learning techniques:

1. **K-Means Clustering**
2. **Decision Tree Classification**

The user can upload a student dataset in **CSV format**. The application automatically performs basic preprocessing and provides an interactive interface for analyzing the dataset.

K-Means is used to group students based on selected numerical characteristics, while Decision Tree Classification is used to predict the student's performance category.

The project combines concepts of:

- Data Mining
- Machine Learning
- Data Preprocessing
- Clustering
- Classification
- Model Evaluation
- Data Visualization
- Web Application Development

---

# ❓ Problem Statement

Student performance depends on several factors such as:

- Attendance
- Study time
- Previous academic performance
- Quiz marks
- Midterm marks
- Laboratory attendance
- Final examination marks
- Other academic factors

Analyzing these factors manually can be difficult when the dataset contains many students.

The objective of this project is to develop an interactive system that can:

- Accept student data through a CSV file
- Process and clean the dataset
- Analyze student groups
- Identify patterns using clustering
- Categorize students according to performance
- Predict performance using a classification model
- Display the results in an understandable format

---

# 🎯 Motivation

Educational institutions generate a large amount of student-related data.

Machine Learning can be used to analyze this data and discover useful patterns.

For example, students may be grouped according to similar:

- Attendance
- Study habits
- Academic marks
- Previous grades
- Laboratory participation

Classification can then be used to predict the performance category of students.

This project demonstrates how Data Mining and Machine Learning techniques can be applied to an educational problem.

---

# 🎯 Objectives

The major objectives of this project are:

- To develop a student performance analysis system.
- To allow CSV datasets to be uploaded by the user.
- To perform basic data preprocessing.
- To identify groups of similar students using K-Means.
- To classify students into performance categories.
- To use Decision Tree Classification for prediction.
- To calculate classification accuracy.
- To generate a classification report.
- To provide an easy-to-use Streamlit interface.
- To demonstrate practical implementation of Data Mining concepts.

---

# ⭐ Key Features

The application provides the following features:

### 📂 CSV Dataset Upload

The user can upload a student dataset in CSV format.

The application accepts:

```text
.csv
````

Only CSV files are used as input.

---

### 📊 Dataset Preview

After uploading the dataset, the application displays:

- First few rows
- Number of rows
- Number of columns
- Number of missing values

---

### 🧹 Data Preprocessing

The application performs basic preprocessing such as:

- Removing duplicate rows
- Removing completely empty columns
- Handling missing numerical values
- Removing an empty `assignments_submitted` column if present
- Creating `final_marks` if it is not present
- Creating the `Performance` target category

---

### 🔵 K-Means Clustering

The user can:

- Select numerical features
- Choose the number of clusters
- Run K-Means
- View the cluster assigned to each student
- View cluster-wise feature averages

---

### 🌳 Decision Tree Classification

The application uses a Decision Tree to predict:

```text
Performance
```

The performance categories are:

```text
Fail
Average
Good
Excellent
```

---

### 📈 Model Evaluation

The classification section displays:

- Accuracy
- Features used
- Precision
- Recall
- F1-score
- Support

---

# 🤖 Machine Learning Techniques

The project uses two main Machine Learning techniques.

| Technique     | Type                  | Purpose                      |
| ------------- | --------------------- | ---------------------------- |
| K-Means       | Unsupervised Learning | Group similar students       |
| Decision Tree | Supervised Learning   | Classify student performance |

---

# 📊 Dataset

The application is designed to work with **student datasets in CSV format**.

The dataset should contain numerical student-related attributes for effective analysis.

Example datasets can contain attributes related to:

- Student information
- Attendance
- Study time
- Previous grades
- Quiz marks
- Midterm marks
- Laboratory attendance
- Final marks

The application is not restricted to one fixed dataset structure as long as suitable numerical columns are available.

---

# 📝 Dataset Attributes

Depending on the uploaded dataset, possible attributes include:

| Attribute            | Description                            |
| -------------------- | -------------------------------------- |
| `student_id`         | Unique student identifier              |
| `age`                | Age of the student                     |
| `study_time_hours`   | Daily/weekly study time                |
| `attendance_percent` | Attendance percentage                  |
| `previous_grade`     | Previous academic grade                |
| `quiz1_marks`        | First quiz marks                       |
| `quiz2_marks`        | Second quiz marks                      |
| `quiz3_marks`        | Third quiz marks                       |
| `midterm_marks`      | Midterm examination marks              |
| `final_marks`        | Final examination marks                |
| `lectures_attended`  | Number of lectures attended            |
| `total_lectures`     | Total number of lectures               |
| `labs_attended`      | Number of laboratory sessions attended |
| `total_lab_sessions` | Total laboratory sessions              |

The exact columns depend on the uploaded CSV file.

---

# 🧹 Data Preprocessing

Before applying Machine Learning algorithms, the dataset is preprocessed.

## 1. Remove Duplicate Rows

Duplicate records are removed using:

```python
df.drop_duplicates()
```

This prevents repeated records from affecting the analysis.

---

## 2. Remove Completely Empty Columns

Columns containing only missing values are removed.

```python
df.dropna(axis=1, how="all")
```

---

## 3. Handle Empty Assignment Column

If the dataset contains:

```text
assignments_submitted
```

and the entire column is empty, it is removed.

---

## 4. Handle Missing Numerical Values

Numerical missing values are converted to `NaN` and replaced using the median value of the corresponding column.

This helps the Machine Learning algorithms work with incomplete datasets.

---

## 5. Generate Final Marks

If the uploaded dataset does not contain:

```text
final_marks
```

the application creates a final marks column using randomly generated values between 40 and 100.

A fixed random seed is used so that the generated values remain reproducible.

---

# 📌 Final Marks

The application uses:

```text
final_marks
```

as the basis for creating the student's performance category.

The marks are divided into four categories.

| Marks  | Performance |
| ------ | ----------- |
| 0–40   | Fail        |
| 41–60  | Average     |
| 61–75  | Good        |
| 76–100 | Excellent   |

The categories are created using:

```python
pd.cut()
```

---

# 🎓 Performance Categories

The target variable used for classification is:

```text
Performance
```

It contains four possible categories:

### ❌ Fail

Students with very low final marks.

### 🟡 Average

Students with moderate final marks.

### 🟢 Good

Students with good academic performance.

### ⭐ Excellent

Students with high final marks.

These categories are used as the target variable for Decision Tree Classification.

---

# 🔵 K-Means Clustering

## What is K-Means?

K-Means is an **unsupervised Machine Learning algorithm** used to divide data into groups called clusters.

In this project, K-Means is used to group students having similar characteristics.

For example, students may be grouped according to:

- Attendance
- Study time
- Previous grades
- Quiz marks
- Midterm marks

---

## Working of K-Means

The basic process is:

1. Select the numerical features.
2. Select the number of clusters `K`.
3. Scale the selected features.
4. Initialize cluster centroids.
5. Assign students to the nearest centroid.
6. Recalculate the centroids.
7. Repeat until the clusters stabilize.

---

## Feature Scaling

Before applying K-Means, the selected features are standardized using:

```python
StandardScaler()
```

This is important because different features may have different ranges.

For example:

```text
Attendance = 0–100
Study Hours = 0–10
Marks = 0–100
```

Scaling prevents features with larger numerical ranges from dominating the clustering process.

---

## Number of Clusters

The application allows the user to select:

```text
K = 2 to 5
```

The selected number of clusters is then passed to:

```python
KMeans()
```

---

## Cluster Output

After clustering, a new column is added:

```text
Cluster
```

Example:

| Student   | Performance | Cluster |
| --------- | ----------- | ------- |
| Student 1 | Good        | 0       |
| Student 2 | Excellent   | 1       |
| Student 3 | Average     | 2       |

The cluster numbers themselves do not represent performance levels.

They simply represent groups created by the K-Means algorithm.

---

# 🌳 Decision Tree Classification

## What is Decision Tree?

A Decision Tree is a supervised Machine Learning algorithm used for classification and prediction.

It represents decisions in the form of a tree.

The tree contains:

- Root node
- Decision nodes
- Branches
- Leaf nodes

---

## Purpose in This Project

The Decision Tree predicts the:

```text
Performance
```

category of a student.

Possible predictions are:

```text
Fail
Average
Good
Excellent
```

---

## Features Used

The model automatically selects available numerical features.

The following are excluded from the classification input:

```text
final_marks
student_id
Student_ID
id
ID
```

`final_marks` is excluded because it is directly used to create the target `Performance`.

This prevents direct target leakage.

---

# ⚠️ Target Leakage

Target leakage occurs when information used to create the target is also directly provided to the model as an input feature.

In this project:

```text
final_marks → Performance
```

Therefore, `final_marks` is excluded from the Decision Tree input features.

This makes the classification setup more meaningful because the model predicts the performance category from other available student attributes.

---

# 📚 Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to build the Decision Tree.

The testing data is used to evaluate how the model performs on unseen data.

The split uses:

```python
random_state=42
```

for reproducibility.

Stratified splitting is attempted so that the performance classes are represented appropriately in training and testing data.

---

# 📈 Model Evaluation

The Decision Tree is evaluated using several metrics.

## Accuracy

Accuracy represents the proportion of correctly classified samples.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

The application displays accuracy as a percentage.

Example:

```text
Accuracy: 85.00%
```

---

## Precision

Precision measures how many of the samples predicted as a particular class actually belong to that class.

```text
Precision = True Positives / (True Positives + False Positives)
```

---

## Recall

Recall measures how many actual samples of a class were correctly identified.

```text
Recall = True Positives / (True Positives + False Negatives)
```

---

## F1-Score

F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Support

Support represents the number of actual samples belonging to each class in the test dataset.

---

# 🔄 Project Workflow

The complete workflow of the application is:

```text
Upload CSV Dataset
        ↓
Read Dataset
        ↓
Data Preprocessing
        ↓
Handle Missing Values
        ↓
Create Final Marks if Required
        ↓
Create Performance Category
        ↓
        ┌─────────────────────┐
        │                     │
        ↓                     ↓
K-Means Clustering     Decision Tree
        ↓                     ↓
Student Groups         Performance Prediction
        ↓                     ↓
Cluster Summary        Accuracy & Report
        └─────────────┬───────┘
                      ↓
                Display Results
```

---

# 🏗️ System Architecture

The application consists of the following major components:

```text
                 User
                  |
                  ↓
          Streamlit Web UI
                  |
                  ↓
           CSV File Upload
                  |
                  ↓
          Data Preprocessing
                  |
        ┌─────────┴─────────┐
        ↓                   ↓
 K-Means Clustering   Decision Tree
        ↓                   ↓
 Student Clusters     Classification
        ↓                   ↓
 Cluster Summary      Evaluation Metrics
        └─────────┬─────────┘
                  ↓
            Results Display
```

---

# 🖥️ Application Interface

The Streamlit application contains the following sections:

## 1. Dataset Upload

The user uploads a CSV file.

```text
Upload Student CSV Dataset
```

---

## 2. Dataset Preview

The application displays the first few rows of the dataset.

---

## 3. Dataset Information

The application shows:

- Number of rows
- Number of columns
- Missing values

---

## 4. Final Marks & Performance

The application displays:

```text
final_marks
Performance
```

---

## 5. K-Means Clustering

The user can:

- Select clustering features
- Select number of clusters
- Run K-Means

---

## 6. Cluster Summary

The application displays average values of the selected features for each cluster.

---

## 7. Decision Tree Classification

The user can run the classification model.

The application displays:

- Accuracy
- Features used
- Classification report
- Performance categories

---

# 🛠️ Technologies Used

## Programming Language

**Python**

Python is used for:

- Data processing
- Machine Learning
- Model training
- Web application development

---

## Web Framework

**Streamlit**

Streamlit is used to create the interactive web interface.

---

## Machine Learning

**Scikit-learn**

Scikit-learn is used for:

- K-Means
- Decision Tree
- StandardScaler
- Train-test split
- Accuracy
- Classification report

---

## Data Processing

**Pandas**

Pandas is used for:

- Reading CSV files
- Data cleaning
- Data manipulation
- Data analysis

---

## Numerical Computing

**NumPy**

NumPy is used for:

- Numerical operations
- Random value generation
- Dataset processing

---

# 📦 Python Libraries

The project uses the following main libraries:

```text
streamlit
pandas
numpy
scikit-learn
```

## Performance Output

Displays:

```text
Final Marks
Performance Category
```

---

## Clustering Output

Displays:

```text
Student ID
Final Marks
Performance
Cluster
```

when the corresponding ID field is available.

---

## Cluster Summary

Displays the mean values of selected features for every cluster.

Example:

```text
Cluster    Attendance    Study Time    Previous Grade
0          75.20         3.40          68.50
1          91.40         5.20          82.30
2          62.80         2.10          55.60
```

---

## Classification Output

The classification section displays:

```text
Accuracy
```

and a classification report containing:

```text
Precision
Recall
F1-score
Support
```

---

# 🧠 Data Mining Concepts Used

This project demonstrates important Data Mining concepts.

## 1. Data Preprocessing

Data is cleaned before applying algorithms.

---

## 2. Clustering

K-Means is used to discover groups of similar students.

---

## 3. Classification

Decision Tree is used to classify students according to performance.

---

## 4. Feature Selection

Relevant numerical features are selected for Machine Learning.

---

## 5. Feature Scaling

StandardScaler is used before K-Means.

---

## 6. Model Evaluation

The classification model is evaluated using accuracy and classification metrics.

---

# 🎓 Academic Concepts Demonstrated

The project demonstrates concepts from:

- Data Mining
- Machine Learning
- Data Warehousing
- Statistics
- Data Preprocessing
- Unsupervised Learning
- Supervised Learning
- Classification
- Clustering
- Model Evaluation
- Python Programming
- Web Application Development

---

# 💡 Use Cases

The project can be used for educational analysis such as:

### Student Group Analysis

Identify groups of students with similar academic characteristics.

### Attendance Analysis

Study relationships between attendance and academic characteristics.

### Academic Performance Analysis

Analyze student performance using multiple numerical attributes.

### Performance Classification

Classify students into different performance categories.

### Educational Data Mining

Demonstrate how Machine Learning can be applied to educational datasets.

---

# ✅ Advantages

- Simple and easy-to-use interface.
- CSV-based input.
- No fixed dataset is required.
- Supports different student datasets containing suitable numerical features.
- Uses both supervised and unsupervised Machine Learning.
- Interactive feature selection.
- Interactive cluster selection.
- Displays understandable results.
- Provides classification metrics.
- Suitable for academic demonstration.
- Can be run locally.
- Easy to modify and extend.

---

# ⚠️ Limitations

The current version has some limitations.

### 1. CSV Input Only

The application currently accepts CSV datasets only.

---

### 2. Numerical Features

K-Means requires numerical features.

Datasets containing mostly categorical attributes may require additional preprocessing.

---

### 3. Generated Final Marks

If the uploaded dataset does not contain `final_marks`, the application generates random values between 40 and 100.

Therefore, these generated values are intended for demonstration rather than real academic evaluation.

---

### 4. Number of Clusters

The current interface allows:

```text
K = 2 to 5
```

---

### 5. Decision Tree Configuration

The Decision Tree uses:

```python
max_depth=5
```

This is a fixed configuration in the current implementation.

---

### 6. No Database

The current application does not store uploaded datasets or results in a database.

---

# 🔮 Future Enhancements

The project can be extended in the future.

Possible improvements include:

- Add more Machine Learning algorithms.
- Add visualization of clusters.
- Add confusion matrix.
- Add feature importance graphs.
- Add downloadable prediction results.
- Add interactive charts.
- Add database support.
- Add authentication.
- Add student-wise analysis.
- Add attendance percentage calculation.
- Add automatic selection of the best number of clusters.
- Add more advanced preprocessing.
- Add support for categorical variables.
- Add model comparison.
- Add historical student performance analysis.
- Add prediction for individual students.

---

# 🧪 Testing

The application can be tested using different CSV datasets.

Testing includes:

## Test Case 1 — Valid CSV

Upload a valid student CSV file.

Expected result:

```text
Dataset loaded successfully!
```

---

## Test Case 2 — Dataset Preview

Upload a CSV file containing student records.

Expected result:

The application displays the first 10 rows.

---

## Test Case 3 — Missing Values

Upload a dataset containing missing numerical values.

Expected result:

Missing numerical values are handled using median replacement during Machine Learning processing.

---

## Test Case 4 — K-Means

Select at least two numerical features and run K-Means.

Expected result:

Clusters and cluster summary are displayed.

---

## Test Case 5 — Classification

Run Decision Tree Classification.

Expected result:

The application displays:

```text
Accuracy
Classification Report
Features Used
```

---

## Test Case 6 — Invalid Clustering Features

Select fewer than two features.

Expected result:

The application displays an error message requesting at least two features.

---

## Test Case 7 — Insufficient Classification Features

If the uploaded dataset does not contain enough suitable numerical features, the application displays a classification error.

---

# 🚨 Error Handling

The application contains basic error handling using `try-except`.

For example:

```python
try:
    ...
except Exception as e:
    st.error(f"Dataset Error: {e}")
```

Errors related to:

- Dataset loading
- Clustering
- Classification

are displayed directly in the Streamlit interface.

This prevents the entire application from stopping unexpectedly due to common input problems.

---

# 🔐 Data Considerations

The application is designed as an academic demonstration system.

Users should avoid uploading sensitive personal student information unless appropriate authorization and data-protection requirements are satisfied.

For demonstration purposes, anonymized or synthetic datasets are preferable.

---

# 📌 Important Implementation Details

The application uses:

```text
Random State = 42
```

for reproducibility.

K-Means uses:

```python
n_init=10
```

The Decision Tree uses:

```python
max_depth=5
```

The classification test size is:

```text
20%
```

The training size is:

```text
80%
```

---

# 🔬 Machine Learning Pipeline

The complete Machine Learning pipeline is:

```text
CSV Dataset
     ↓
Data Loading
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Feature Selection
     ↓
     ├───────────────────────┐
     ↓                       ↓
Standardization          Train-Test Split
     ↓                       ↓
K-Means                  Decision Tree
     ↓                       ↓
Clusters                Prediction
     ↓                       ↓
Cluster Summary         Evaluation
```

---

# 📊 Example Performance Categories

The project creates the following categories from final marks:

```text
0–40       → Fail
41–60      → Average
61–75      → Good
76–100     → Excellent
```

These categories are generated programmatically from the `final_marks` column.

---

# 🔍 Example K-Means Features

Depending on the uploaded dataset, possible clustering features include:

```text
attendance_percent
study_time_hours
previous_grade
quiz1_marks
quiz2_marks
quiz3_marks
midterm_marks
lectures_attended
labs_attended
```

The user can select the available numerical features through the Streamlit interface.

---

# 🌳 Example Decision Tree Features

The classification model automatically identifies numerical features available in the uploaded dataset.

Features such as:

```text
attendance
study time
previous grade
quiz marks
midterm marks
lecture attendance
lab attendance
```

can be used when available.

The target is:

```text
Performance
```

---

# 📚 Why Two Algorithms?

The project uses both clustering and classification to demonstrate two different Machine Learning approaches.

### K-Means

K-Means does not require predefined target labels.

It discovers groups based on similarities in the data.

### Decision Tree

Decision Tree uses a known target variable and learns patterns that can be used to classify new observations.

Using both techniques demonstrates:

```text
Unsupervised Learning + Supervised Learning
```

within the same project.

---

# 🧩 Project Components

The complete application consists of:

```text
1. Dataset Upload
2. Data Preprocessing
3. Performance Generation
4. K-Means Clustering
5. Cluster Analysis
6. Decision Tree Classification
7. Model Evaluation
8. Streamlit User Interface
```

---

# 🎓 Academic Relevance

This project is suitable for demonstrating practical implementation of **Data Mining and Machine Learning** concepts.

It covers:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Transformation
      ↓
Feature Selection
      ↓
Clustering
      ↓
Classification
      ↓
Evaluation
      ↓
Result Interpretation
```

The project demonstrates how theoretical Machine Learning algorithms can be integrated into a working web-based application.

---

# 👥 Team

This project is developed as an academic project by a **two-member team**.

The work includes:

- Dataset preparation
- Data preprocessing
- Machine Learning implementation
- Streamlit development
- Testing
- Documentation

---

# 🚀 Project Goal

The overall goal of this project is to provide a simple and interactive system that demonstrates how student-related data can be analyzed using Machine Learning.

The application combines:

```text
Student Data
      +
Data Preprocessing
      +
K-Means Clustering
      +
Decision Tree Classification
      +
Model Evaluation
      =
Student Attendance & Performance Analysis
```

---

# 🏁 Conclusion

The **Student Attendance & Performance Analysis** project demonstrates the practical use of Data Mining and Machine Learning techniques for analyzing educational data.

The application accepts student data through a CSV file, performs basic preprocessing, creates performance categories, groups students using K-Means Clustering, and predicts performance using Decision Tree Classification.

The project provides an interactive Streamlit interface so that users can upload datasets, select features, run Machine Learning algorithms, and view the results.

Overall, the project provides a practical implementation of:

- Data Preprocessing
- K-Means Clustering
- Decision Tree Classification
- Feature Selection
- Model Evaluation
- Educational Data Mining
- Streamlit Web Development

---

# 📄 License

This project is developed for **academic and educational purposes**.

It is intended to demonstrate the practical implementation of Data Mining and Machine Learning concepts.

```
```