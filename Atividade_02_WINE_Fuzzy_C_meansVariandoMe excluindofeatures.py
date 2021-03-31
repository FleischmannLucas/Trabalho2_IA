# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:13:54 2021

@author: Lucas Gava
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import numpy as np
import collections
from sklearn.metrics import classification_report      #relatorio de acertividade do metodo
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Carrega o wine dataset em wine 
wine = load_wine()

# excluindo features
f1 = wine.data[:,0]
f2 = wine.data[:,1]
f3 =  wine.data[:,2]
f4 = wine.data[:,3]
f5 = wine.data[:,4]
f6 =  wine.data[:,5]
f7 = wine.data[:,6]
f8 = wine.data[:,7]
f9 =  wine.data[:,8]
f10 = wine.data[:,9]
f11 = wine.data[:,10]
f12 =  wine.data[:,11]
f13 = wine.data[:,11]

wine.data = np.column_stack(( f1, f2,f6,f7,f8,f10,f11,f12,f13 ))


#Divisao dos grupos de treino e teste com quantidades iguais de cada grupo

classe1=wine.data[:58,:]
classe1y=wine.target[:58]
classe1train, classe1test, classe1ytrain, classe1ytest=train_test_split(classe1, classe1y, random_state=5)

classe2=wine.data[59:129,:]
classe2y=wine.target[59:129]
classe2train, classe2test, classe2ytrain, classe2ytest=train_test_split(classe2, classe2y, random_state=5)

classe3=wine.data[130:177,:]
classe3y=wine.target[130:177]
classe3train, classe3test, classe3ytrain, classe3ytest=train_test_split(classe3, classe3y, random_state=5)

Xtest = np.concatenate((classe1test, classe2test, classe3test), axis=0)
Xtrain = np.concatenate((classe1train, classe2train, classe3train), axis=0)
Ytest= np.concatenate((classe1ytest, classe2ytest, classe3ytest), axis=0)
Ytrain= np.concatenate((classe1ytrain, classe2ytrain, classe3ytrain), axis=0)

Xtrain = Xtrain.transpose()    #algoritimo precisa da entrada transposta
Xtest = Xtest.transpose()




#Carrega o iris dataset em iris 


ncenters = 3
acc = []

for i in range(2,100,1):

    m=i
#Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    Xtrain, ncenters, m, error=0.001, maxiter=2000, init=None)
    cluster_membership = np.argmax(u, axis=0)

#fazendo a previsao para o grupo de test

    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    Xtest, cntr, m, error=0.001, maxiter=2000)




#corrigindo o problema da saida estar com as labels trocadas: normalizacao das labels de saida
    
    Z=np.zeros((45,))

    x= np.argmax(u, axis=0)

    c=collections.Counter(x[:15])
    c1=np.array(c.most_common())

    c=collections.Counter(x[15:33])
    c2=np.array(c.most_common())

    c=collections.Counter(x[33:45])
    c3=np.array(c.most_common())



    for i in range(0,45,1):
    # print (i)
        if (x[i]==c1[0,0]):
            Z[i] = 0
        if (x[i]==c2[0,0]):
            Z[i] = 1
        if (x[i]==c3[0,0]):
            Z[i] = 2
 
#saida correta independente do caso, Nice

    
    acc.append(accuracy_score( Z , Ytest ))

    print(m)
    


    

fig0, ax0 = plt.subplots(figsize=(15, 8))
ax0.plot(np.r_[2:100], acc)
ax0.set_xlabel("Valor M")
ax0.set_ylabel("Acurace total")


"""

# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = escolher os dados a serem testados

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)
"""