import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import MLPythonKsiazka as P

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
print(df.tail())

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
x = df.iloc[0:100, [0,2]].values
"""
plt.scatter(x[:50,0], x[:50,1], color='red', marker = 'o', label='Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker = 'x', label='Versicolor')
plt.xlabel('Długość działki w cm')
plt.ylabel('Długość płatka w cm')
plt.show()
"""
#perceptron training 
ppn=P.Perceptron(eta=0.1, n_iter=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')
plt.show()