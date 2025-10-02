import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("50_Startups.csv")

print("Problem 2: Profit Prediction")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

print("Correlation:")
print(df.corr(numeric_only=True))

y = df["Profit"]
X = pd.get_dummies(df.drop(columns=["Profit"]), drop_first=True)

chosen = ["R&D Spend","Marketing Spend"]
print("Chosen explanatory vars:", chosen)

for v in chosen:
    plt.scatter(df[v], y)
    plt.xlabel(v)
    plt.ylabel("Profit")
    plt.title(v + " vs Profit")
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X[chosen], y, test_size=0.2, random_state=1)
lr = LinearRegression().fit(X_train, y_train)
train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)

print("Train R2:", r2_score(y_train, train_pred))
print("Test R2:", r2_score(y_test, test_pred))
print("Train RMSE:", mean_squared_error(y_train, train_pred, squared=False))
print("Test RMSE:", mean_squared_error(y_test, test_pred, squared=False))

"""
Here I explored the startup dataset and looked at correlations. 
R&D spend and marketing spend came out as the strongest predictors. 
I plotted them to check the relation with profit and it looked linear. 
Then I split into training and test sets and trained a linear regression. 
I measured R2 and RMSE on both sets. The model gave strong results which means the choice was correct.
"""
