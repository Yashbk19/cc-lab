import numpy as np
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([[1.2],[1.8],[2.6],[3.2],[3.8]])
mean_x=np.mean(x)
mean_y=np.mean(y)   
print(mean_x)
print(mean_y)

xm_val=x-mean_x
print(xm_val)

ym_val=y-mean_y
print(ym_val)

sqxm_val=(xm_val**2)
print(sqxm_val)

xs=sum(sqxm_val)
print(xs)

mul = xm_val*ym_val
print(mul)

m=sum(mul)/xs
print(m)

c=mean_y-(m*mean_x)
print(c)

x_new=12
y_new=(m*x_new)+c
print(y_new)

import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1.2,1.8,2.6,3.2,3.8]
plt.scatter(x,y,color='blue')
plt.scatter(x,y,color='red')
plt.plot(x,y,color='green')
plt.show()


import numpy as np
from sklearn.linear_model import LinearRegression
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([[1.2],[1.8],[2.6],[3.2],[3.8]])
reg=LinearRegression().fit(x,y)
print(reg.coef_)
print(reg.intercept_)

print(reg.predict([[12]]))
