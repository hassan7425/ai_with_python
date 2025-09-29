import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
sizes=[500,1000,2000,5000,10000,15000,20000,50000,100000]
prob={2:1/36,3:2/36,4:3/36,5:4/36,6:5/36,7:6/36,8:5/36,9:4/36,10:3/36,11:2/36,12:1/36}
x=np.arange(2,13)
y=np.array([prob[i] for i in x])
for n in sizes:
 d1=np.random.randint(1,7,n)
 d2=np.random.randint(1,7,n)
 s=d1+d2
 h,e=np.histogram(s,range(2,14))
 p=h/n
 plt.bar(e[:-1],p)
 plt.plot(x,y,'ro-')
 plt.title(f'n={n}')
 filename=f'vA_ex1_{n}.png'
 plt.savefig(filename)
 print("Image saved as",filename)
 plt.close()
print("done vA ex1")


# answer 4 (What I observe): At first the bars jump around, but with big n it looks like the true probabilities.
# Answer 5: The odd results balance out, pulling everything back near the average = regression to mean.
