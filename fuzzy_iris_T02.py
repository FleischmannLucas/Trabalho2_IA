# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:38:12 2021

@author: Gustavo
"""

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Dimensionamento das variaveis 
Lpetala = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'Lpetala')
Cpetala = ctrl.Antecedent(np.arange(1, 6.4, 0.1), 'Cpetala')
tipo = ctrl.Consequent(np.arange(0, 3, 1), 'tipo')



Lpetala['P']= fuzz.trimf(Lpetala.universe, [0, 0, 1])
Lpetala['M']= fuzz.trimf(Lpetala.universe, [0.5, 1.5, 2.2])
Lpetala['G']= fuzz.trimf(Lpetala.universe, [1.5, 3, 3])

Lpetala.view() #grafico fuzzy imput largura da petala

Cpetala['p']= fuzz.trimf(Cpetala.universe, [1, 1, 2.5])
Cpetala['m']= fuzz.trimf(Cpetala.universe, [2.5, 3.5, 4.8])
Cpetala['g']= fuzz.trimf(Cpetala.universe, [3.5, 6.3, 6.3])

Cpetala.view()  #grafico fuzzy imput comprimento da petala

tipo['cetosa'] = fuzz.trimf(tipo.universe, [0, 0, 1])
tipo['versicola'] = fuzz.trimf(tipo.universe, [0.5, 1, 1.5])
tipo['virginica'] = fuzz.trimf(tipo.universe, [1.5, 2, 2])

tipo.view()  #grafico fuzzy saida tipo


#regras de classificaçao
rule1 = ctrl.Rule(Lpetala['P'] | Cpetala['p'] , tipo['cetosa'])
rule2 = ctrl.Rule(Lpetala['M'] & Cpetala['m'] , tipo['versicola'])
rule3 = ctrl.Rule(Lpetala['G'] & Cpetala['g'] , tipo['virginica'])
rulr4 = ctrl.Rule(Lpetala['M'] |Cpetala['m']  , tipo['versicola'])
rule5 = ctrl.Rule(Lpetala['G'] | Cpetala['g'] , tipo['virginica'])
rule6 = ctrl.Rule(Lpetala['P'] & Cpetala['p'] , tipo['cetosa'])


#comandos de simulçao
classe_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rulr4, rule5,rule6 ])
classe = ctrl.ControlSystemSimulation(classe_ctrl)

#entrada de dados
iris = load_iris() 
Y=[]


for i in range(0,150,1):

    a = iris.data[i,[3]]
    b = iris.data[i,[2]]
    
    LargPetala = float(np.asarray(a))  
    CompPetala = float(np.asarray(b))
    
    classe.input['Lpetala'] = LargPetala
    classe.input['Cpetala'] = CompPetala

    classe.compute()
    if (classe.output['tipo']<0.8):
        Y.append(0)
    if (classe.output['tipo']<1.1):
            if (classe.output['tipo']>0.80):
                Y.append(1)
                
    if (classe.output['tipo']>1.1):
                    Y.append(2)
    
    
    
    x = iris.target

    
print(classe.output['tipo'])


Lpetala.view(sim=classe)



Cpetala.view(sim=classe)


tipo.view(sim=classe)

print(accuracy_score( Y, x )) 

print(classification_report(Y, x))













