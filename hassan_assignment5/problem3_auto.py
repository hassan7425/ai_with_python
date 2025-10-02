import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

auto = pd.read_csv("Auto.csv", na_values=["?","NA"]).dropna()

y = auto["mpg"]
X = auto.drop(columns=["mpg","name","origin"])
X = X.apply(pd.to_numeric, errors="coerce").dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

alphas = [0.001,0.01,0.1,1,10,100]
ridge_scores = []
lasso_scores = []

ridge = Pipeline([("scale", StandardScaler()),("ridge", Ridge())])
lasso = Pipeline([("scale", StandardScaler()),("lasso", Lasso(max_iter=10000))])

for a in alphas:
    ridge.set_params(ridge__alpha=a)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))
    lasso.set_params(lasso__alpha=a)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

plt.plot(alphas, ridge_scores, label="Ridge")
plt.plot(alphas, lasso_scores, label="Lasso")
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("R2")
plt.legend()
plt.show()

print("Problem 3: Car mpg")
print("Best Ridge alpha:", alphas[np.argmax(ridge_scores)], "with R2:", max(ridge_scores))
print("Best Lasso alpha:", alphas[np.argmax(lasso_scores)], "with R2:", max(lasso_scores))

"""
I used Auto.csv and set mpg as the target. 
I dropped name and origin since they are not numeric features to use. 
I trained Ridge and Lasso with a range of alpha values and stored the test R2 scores. 
The results were plotted on a log scale. 
I picked the alpha that gave the best test score for each model. 
Ridge kept good performance for bigger range while Lasso dropped faster with higher alpha.
"""
