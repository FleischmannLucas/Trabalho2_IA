# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:44:40 2021

@author: Lucas Gava
"""



from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import datasets
import seaborn as sns
import pandas as pd

#Carrega o Wine dataset em iris 
wine = load_wine()

#Divisao dos grupos de treino e teste com quantidades iguais de cada grupo
wine_df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
wine_df['label']=wine.target
wine_df['species'] = pd.Categorical.from_codes(wine.target, wine.target_names)

sns.pairplot(wine_df[['alcohol','malic_acid','ash','alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline','species']], hue="species")


wine_df.head()





# classe1=wine.data[:58,:]
# classe1y=wine.target[:58]
# classe1train, classe1test, classe1ytrain, classe1ytest=train_test_split(classe1, classe1y, random_state=5)

# classe2=wine.data[59:129,:]
# classe2y=wine.target[59:129]
# classe2train, classe2test, classe2ytrain, classe2ytest=train_test_split(classe2, classe2y, random_state=5)

# classe3=wine.data[130:177,:]
# classe3y=wine.target[130:177]
# classe3train, classe3test, classe3ytrain, classe3ytest=train_test_split(classe3, classe3y, random_state=5)

# Xtest = np.concatenate((classe1test, classe2test, classe3test), axis=0)
# Xtrain = np.concatenate((classe1train, classe2train, classe3train), axis=0)
# Ytest= np.concatenate((classe1ytest, classe2ytest, classe3ytest), axis=0)
# Ytrain= np.concatenate((classe1ytrain, classe2ytrain, classe3ytrain), axis=0)


# #iniciação do classificador SVM
# clf = svm.SVC(C=1.0)
# clf.fit(Xtrain,Ytrain)


# print(clf.predict(Xtest)) #imprime os resultados

# Z = clf.predict(Xtest)

# print(accuracy_score(Z, Ytest))     #imprime acuracia total

# print(classification_report(Z, Ytest))   #imprime tabela de acuracia


# print(accuracy_score(clf.predict(classe1test), classe1ytest))  #calcula a acuracia para tipo 1
# print(accuracy_score(clf.predict(classe2test), classe2ytest))  #calcula a acuracia para tipo 2 
# print(accuracy_score(clf.predict(classe3test), classe3ytest))  #calcula a acuracia para tipo 3


