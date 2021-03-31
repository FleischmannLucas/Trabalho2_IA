# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:13:54 2021

@author: Lucas Gava
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris
import numpy as np
import collections
from sklearn.metrics import classification_report      #relatorio de acertividade do metodo
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Carrega o iris dataset em iris 
iris = load_iris()



#divisao do data set entra grupo de treino e test
cetosa= iris.data[:50,:]
cetosay= iris.target[:50]
cet=np.array(cetosa)
cety=np.array(cetosay)
cettrain, cettest, cetytrain, cetytest=train_test_split(cet, cety, random_state=5)

versicolor= iris.data[50:100,:]
versicolory= iris.target[50:100]
ver=np.array(versicolor)
very=np.array(versicolory)
vertrain, vertest, verytrain, verytest=train_test_split(ver, very, random_state=5)

virginica= iris.data[100:150,:]
virginicay= iris.target[100:150]
virg=np.array(virginica)
virgy=np.array(virginicay)
virgtrain, virgtest, virgytrain, virgytest=train_test_split(virg, virgy, random_state=5)

Xtest = np.concatenate((cettest, vertest, virgtest), axis=0)
Xtrain = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Ytest = np.concatenate((cetytest, verytest, virgytest), axis=0)
Ytrain = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)

Xtrain = Xtrain.transpose()    #algoritimo precisa da entrada transposta
Xtest = Xtest.transpose()




#Carrega o iris dataset em iris 


ncenters = 3
acc = []

for i in range(2,1000,1):

    m=i
#Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    Xtrain, ncenters, m, error=0.001, maxiter=2000, init=None)
    cluster_membership = np.argmax(u, axis=0)

#fazendo a previsao para o grupo de test

    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    Xtest, cntr, m, error=0.001, maxiter=2000)




#corrigindo o problema da saida estar com as labels trocadas: normalizacao das labels de saida
    
    Z=np.zeros((39,))

    x= np.argmax(u, axis=0)

    c=collections.Counter(x[:13])
    c1=np.array(c.most_common())

    c=collections.Counter(x[13:26])
    c2=np.array(c.most_common())

    c=collections.Counter(x[26:39])
    c3=np.array(c.most_common())



    for i in range(0,38,1):
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
ax0.plot(np.r_[2:1000], acc)
ax0.set_xlabel("Valor M")
ax0.set_ylabel("Acurace total")


