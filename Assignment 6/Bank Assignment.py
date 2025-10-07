import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

"""
Step 0 & 1:
The dataset used is Bank Marketing data from the UCI repository.
I read the CSV file using pandas with the correct delimiter (;).
After reading the file, I will check columns and datatypes to make sure it loaded properly.
"""

# Load the dataset
df = pd.read_csv("bank.csv", delimiter=';')

# Checking the basic info
print("Columns in dataset:", list(df.columns))
print("\nData types of each column:\n", df.dtypes)
print("\nPreview of data:\n", df.head(3))

"""
Step 2:
I will make a new dataframe (df2) which includes only the following columns:
'y', 'job', 'marital', 'default', 'housing', 'poutcome'
"""

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print("\nCreated df2 with selected columns.")
print(df2.sample(5))

"""
Step 3:
Next, I will transform the categorical columns into numeric dummy variables.
I use pandas get_dummies() function to handle all category encodings.
"""

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print("\nConverted categorical data into dummy variables successfully.")
print("df3 shape:", df3.shape)

"""
Step 4:
Now, I will create a correlation heatmap to visualize how variables relate.
"""

corr_matrix = df3.corr(numeric_only=True)

plt.figure(figsize=(11,7))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap of df3")
plt.show()

"""
Observation:
The heatmap shows that correlations are mostly weak between variables,
because dummy variables rarely show high linear correlation.
The target column 'y' has very minimal correlation with other features.
"""

"""
Step 5:
I will define my dependent variable (y) and independent variables (X).
"""

# Convert 'yes'/'no' into numeric values
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

y = df3['y']
X = df3.drop('y', axis=1)

print("\nData split into X (features) and y (target).")
print(f"X shape: {X.shape}, y shape: {y.shape}")

"""
Step 6:
Splitting data into training and test sets using 75/25 ratio.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nTraining and testing data prepared.")

"""
Step 7:
Training the Logistic Regression model on the training data.
"""

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

"""
Step 8:
Evaluating Logistic Regression performance using confusion matrix and accuracy.
"""

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Results:")
print("Confusion Matrix:\n", cm_log)
print(f"Accuracy Score: {acc_log:.4f}")

sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""
Step 9:
Now I will train a K-Nearest Neighbors model with k = 3.
"""

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\nKNN Results (k=3):")
print("Confusion Matrix:\n", cm_knn)
print(f"Accuracy Score: {acc_knn:.4f}")

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - KNN (k=3)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""
Step 10:
Compare both models and mention which one works better for this dataset.
"""

print("\nModel Comparison Summary:")
print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print(f"KNN Accuracy (k=3): {acc_knn:.4f}")

"""
Findings:
In this dataset, Logistic Regression performs a bit better than KNN.
This is because the features are mostly categorical and converted into dummy variables,
which suits Logistic Regression more than distance-based models like KNN.
"""
