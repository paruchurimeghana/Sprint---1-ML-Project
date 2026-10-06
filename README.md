# Sprint-1-ML-Project

# 🏏 IPL Auction Price Prediction — Sprint 1

# 📌 Project Overview

This project focuses on **Data Understanding and Preprocessing** for an IPL player dataset.

The main goal of this Sprint is to convert **raw player data into a clean and model-ready dataset** that can later be used to build a Machine Learning model for predicting **IPL Auction Price**.

### 🎯 Target Variable

**Auction_Price**

## 🚀 Sprint 1: Data Understanding & Preprocessing

### Workflow

Raw Dataset
     ↓
Data Collection & Loading
     ↓
Initial Data Inspection
     ↓
Exploratory Data Analysis
     ↓
Missing Value Handling
     ↓
Duplicate Detection
     ↓
Outlier Detection & Treatment
     ↓
Feature Encoding
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
Model-Ready Dataset

## 📂 Dataset Information

The dataset contains information about IPL players and their performance.

### Main Columns

| Column | Description |
|---|---|
| Name | Player name |
| Age | Player age |
| Country | Player's country |
| Role | Player's playing role |
| Matches | Number of matches played |
| Experience_Years | Years of experience |
| Runs | Total runs |
| Strike_Rate | Player's strike rate |
| Wickets | Number of wickets |
| Economy | Bowling economy rate |
| Average | Player's average |
| Team_Preference | Preferred team |
| Auction_Price | Player's auction price |
| Role_Encoded | Encoded representation of Role |


# 🔹 Step 1: Data Collection & Loading

The dataset was loaded using **Pandas**.


import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.shape)
print(df.columns.tolist())

### Objective

- Load the dataset
- Check dataset size
- Identify available columns

### Insight

The dataset was successfully loaded and the column structure was verified before preprocessing.


# 🔹 Step 2: Initial Data Inspection

The dataset was inspected using:

df.head()
df.tail()
df.info()
df.describe()

### Objective

- Understand the structure of the dataset
- Identify numerical and categorical features
- Check data types
- Understand statistical properties

### Insight

Initial inspection helped identify the different numerical and categorical features that require different preprocessing techniques.


# 🔹 Step 3: Exploratory Data Analysis

EDA was performed to understand the data distribution and relationships between variables.

### Visualizations Used

- Histograms
- Boxplots
- Countplots
- Scatter plots
- Correlation heatmap

Example:

import matplotlib.pyplot as plt

df['Auction_Price'].hist()
plt.xlabel("Auction Price")
plt.ylabel("Frequency")
plt.show()

### Objective

- Understand feature distributions
- Identify unusual values
- Find relationships between features
- Understand important patterns

### Insight

EDA helps understand the dataset before applying preprocessing and Machine Learning algorithms.


# 🔹 Step 4: Missing Value Handling

Missing values were checked using:

print(df.isnull().sum())


If missing values are present, they can be handled using appropriate techniques such as mean, median, mode, or removal.

Example:

df['Age'] = df['Age'].fillna(df['Age'].median())


### Insight

Missing-value treatment ensures that incomplete data does not cause problems during Machine Learning model training.


# 🔹 Step 5: Duplicate Detection

Duplicate records were identified and removed.

print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())


### Insight

Removing duplicate records prevents repeated observations from affecting the analysis and model.


# 🔹 Step 6: Outlier Detection & Treatment

Outliers were analyzed using **Boxplots and IQR**.

Example:

import matplotlib.pyplot as plt

plt.boxplot(df['Auction_Price'])
plt.ylabel('Auction Price')
plt.show()


### IQR Method

Q1 = df['Auction_Price'].quantile(0.25)
Q3 = df['Auction_Price'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Lower limit:", lower)
print("Upper limit:", upper)


### Insight

Outlier analysis helps identify unusually high or low values that may affect statistical analysis and model performance.



# 🔹 Step 7: Feature Encoding

Categorical features were converted into numerical representation.

### Categorical Features

nominal_columns = [
    'Country',
    'Role',
    'Team_Preference'
]

Since these are **nominal categorical variables**, One-Hot Encoding was used.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('nominal',
         OneHotEncoder(
             sparse_output=False,
             handle_unknown='ignore'
         ),
         nominal_columns)
    ],
    remainder='passthrough'
)

### Important

`Role_Encoded` was not required when using One-Hot Encoding for `Role`, because it would duplicate the representation of the same feature.

### Insight

One-Hot Encoding converts categorical values into numerical features without assuming any artificial order between categories.

# 🔹 Step 8: Feature Scaling

Numerical features were scaled using **StandardScaler**.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Important Rule

Training Data → fit_transform()
Testing Data  → transform()

The scaler is fitted only on training data to avoid **data leakage**.

### Insight

Feature scaling brings numerical features to a comparable scale and helps Machine Learning algorithms work more effectively.

# 🔹 Step 9: Train-Test Split

The dataset was divided into training and testing sets.

from sklearn.model_selection import train_test_split

X = df.drop(
    columns=['Auction_Price', 'Name', 'Role_Encoded'],
    errors='ignore'
)

y = df['Auction_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

### Dataset Split

80% → Training Data
20% → Testing Data

### Insight

The training set is used to learn patterns, while the testing set evaluates how well the model performs on unseen data.


# 💾 Cleaned Dataset

After preprocessing, the cleaned dataset was saved for future Machine Learning steps.

df.to_csv("cleaned_dataset.csv", index=False)

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Google Colab**
- **Jupyter Notebook**

The final objective is to build a Machine Learning model that can predict **IPL Player Auction Price**.


### Author

**Paruchuri Meghana**

B.Tech — Computer Science

### Project Focus

**Data Analytics | Machine Learning | Python**
