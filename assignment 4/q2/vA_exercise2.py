import pandas as pd,numpy as np,matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import math
d=pd.read_csv("weight-height.csv")
X=d[['Height']].values
Y=d['Weight'].values
m=LinearRegression().fit(X,Y)
yp=m.predict(X)
rm=math.sqrt(mean_squared_error(Y,yp))
r2=r2_score(Y,yp)
print("eq: W=%.3f*H+%.3f"%(m.coef_[0],m.intercept_))
print("RMSE=%.3f R2=%.3f"%(rm,r2))
xl=np.linspace(X.min(),X.max(),200).reshape(-1,1)
yl=m.predict(xl)
plt.scatter(X,Y,s=6,alpha=.3)
plt.plot(xl,yl,'g',lw=2)
plt.savefig("vA_ex2.png")
print("Image saved as vA_ex2.png")
plt.show()
