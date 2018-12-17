# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:54:31 2018

@author: RC-X550ZE
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from random import randint

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

jumlah_cluster = 3
center = []

jarak_centroid = []
anggota = []
new_center = []
ind = 0
for i in range(0,jumlah_cluster):
    center.append([])
    anggota.append([])
    new_center.append([])

center[0] = X[1]
center[1] = X[58]
center[2] = X[78]
    
for i in range(0,2):  
    for z in range(0, len(X)):
        jarak = []
        for j in range(0,jumlah_cluster):
            jarak.append(np.linalg.norm(center[j] - X[z]))
        jarak_centroid.append(jarak)
        la = np.where(jarak_centroid[z] == min(jarak_centroid[z]))
        anggota[la[0][0]].append(X[z])
    
    for j in range(0,jumlah_cluster):
        anggota[j] = np.asarray(anggota[j])
        new_center[j].append(np.mean(anggota[j],axis=0))
        #print(new_center[j])
    
    
'''
for j in range(0, jumlah_cluster):
    plt.scatter(anggota[j][:,0],anggota[j][:,1])
    plt.scatter(center[j][0],center[j][1], s=50, marker='+')

plt.show()
'''