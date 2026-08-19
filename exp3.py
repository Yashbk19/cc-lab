import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data={
    'Experience':[1,2,3,4,5],
    'TestScore':[50,60,65,70,80],
    'Salary':[20000,25000,30000,35000,40000]
}
df = pd.DataFrame(data)

x=df[['Experience','TestScore']]
y=df['Salary']  
print(x)


model=LinearRegression().fit(x,y)
model.fit(x.values,y.values)
print("Intercept:",model.intercept_)
print("Coefficients:",model.coef_)

new_data=np.array([[6,85]])
predicted = model.predict(new_data)
print("Predicted Salary ", predicted[0])

#Multiple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
data={
    'Experience':[1,2,3,4,5],
    'TestScore':[50,60,65,70,80],
    'Salary':[20000,25000,30000,35000,40000]
}
df = pd.DataFrame(data)
x=df[['Experience','TestScore']]
y=df['Salary']
model=LinearRegression()
model.fit(x,y)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df['Experience'],df['TestScore'],df['Salary'])
exp_range=np.linspace(df['Experience'].min(),df['Experience'].max(),10)
score_range=np.linspace(df['TestScore'].min(),df['TestScore'].max(),10)
exp_grid,score_grid=np.meshgrid(exp_range,score_range)
salary_pred=model.predict(np.c_[exp_grid.ravel(),score_grid.ravel()])
salary_grid=salary_pred.reshape(exp_grid.shape)

ax.plot_surface(exp_grid,score_grid,salary_grid,alpha=0.5)
ax.set_xlabel('Experience')
ax.set_ylabel('Test Score')
ax.set_zlabel('Salary')

plt.title('Multiple Linear Regression:-3D')
plt.show()