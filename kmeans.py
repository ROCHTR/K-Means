# -*- coding: utf-8 -*-
"""
K = 2
Iris Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from random import randint

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

randint(0, len(X))
'''
center0 = X[randint(0, len(X))]
center1  = X[randint(0, len(X))]
center2  = X[randint(0, len(X))]
'''

center0 = X[1]
center1  = X[58]
center2  = X[78]

jarak0 = np.empty((len(X)))
jarak1 = np.empty((len(X)))
jarak2 = np.empty((len(X)))

jarak = np.empty(3)

cluster0 = []
cluster1 = []
cluster2 = []

titik0 = []
titik1 = []
titik2 = []

for j in range(0,100):
    titik0.append(center0)
    titik1.append(center1)
    titik2.append(center2)
    for i in range(0,len(X)):
        jarak0[i] = np.linalg.norm(center0 - X[i])
        jarak1[i] = np.linalg.norm(center1 - X[i])
        jarak2[i] = np.linalg.norm(center2 - X[i])
        jarak[0] = jarak0[i]
        jarak[1] = jarak1[i]
        jarak[2] = jarak2[i]
        la = np.where(jarak == min(jarak))
        if(la[0] == 1):
            cluster1.append(X[i])
        elif(la[0] == 2):
            cluster2.append(X[i])
        elif(la[0] == 0):
            cluster0.append(X[i])
    cluster0= np.asarray(cluster0)
    cluster1= np.asarray(cluster1)
    cluster2= np.asarray(cluster2)
    new0 = np.mean(cluster0,axis=0)
    new1 = np.mean(cluster1,axis=0)
    new2 = np.mean(cluster2,axis=0)
    print(j)
    if (np.array_equal(center0,new0) == True and np.array_equal(center1,new1) == True and np.array_equal(center2,new2 ) == True):
        center0 = new0
        center1  = new1
        center2  = new2
        titik0.append(center0)
        titik1.append(center1)
        titik2.append(center2)
        break
    else:
        center0 = new0
        center1  = new1
        center2  = new2
        cluster0 = []
        cluster1 = []
        cluster2 = []

plt.scatter(cluster0[:,0],cluster0[:,1])
plt.scatter(cluster1[:,0],cluster1[:,1])
plt.scatter(cluster2[:,0],cluster2[:,1])
plt.scatter(center0[0],center0[1], s=50, marker='>')
plt.scatter(center1[0],center1[1], s=50, marker='+')
plt.scatter(center2[0],center2[1], s=50, marker='X')
plt.show()
