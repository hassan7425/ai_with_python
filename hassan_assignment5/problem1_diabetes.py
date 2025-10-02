import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

m_base = LinearRegression().fit(X_train[["bmi","s5"]], y_train)
r2_base = r2_score(y_test, m_base.predict(X_test[["bmi","s5"]]))
rmse_base = mean_squared_error(y_test, m_base.predict(X_test[["bmi","s5"]]), squared=False)

improvement = {}
for col in X.columns:
    if col not in ["bmi","s5"]:
        m = LinearRegression().fit(X_train[["bmi","s5",col]], y_train)
        score = r2_score(y_test, m.predict(X_test[["bmi","s5",col]]))
        improvement[col] = score

best_var = max(improvement, key=improvement.get)
m_best = LinearRegression().fit(X_train[["bmi","s5",best_var]], y_train)
r2_best = r2_score(y_test, m_best.predict(X_test[["bmi","s5",best_var]]))

m_all = LinearRegression().fit(X_train, y_train)
r2_all = r2_score(y_test, m_all.predict(X_test))

print("Problem 1: Diabetes")
print("Base model R2:", round(r2_base,4), "RMSE:", round(rmse_base,2))
print("Best extra variable:", best_var, "with R2:", round(r2_best,4))
print("All variables R2:", round(r2_all,4))

"""
In this task I first trained the regression using bmi and s5. 
Then I checked every other feature one by one and compared the R2 values. 
The variable that gave the best increase was selected. 
Finally I also tested the model with all features together. 
It shows that adding one more good feature helps, and with all features the score improves more, 
though it is not always a big jump.
"""
