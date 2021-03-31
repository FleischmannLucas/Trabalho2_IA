# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:53:49 2021

@author: Gustavo
"""

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_wine
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Dimensionamento das variaveis 
collor = ctrl.Antecedent(np.arange(1, 15, 0.1), 'Coloração')
flavinoid = ctrl.Antecedent(np.arange(0, 6.01, 0.01), 'Flavinoides')
tipo = ctrl.Consequent(np.arange(0, 3, 1), 'tipo')


collor['rosa'] = fuzz.trimf(collor.universe, [1, 1, 4])
collor['vermelho_claro'] = fuzz.trimf(collor.universe, [3.5, 6, 10.5])
collor['vermelho_escuro'] = fuzz.trimf(collor.universe, [9.5, 12.5, 14.9])

collor.view()

flavinoid['baixo'] = fuzz.trimf(flavinoid.universe, [0, 0.6, 1.2])
flavinoid['médio'] = fuzz.trimf(flavinoid.universe, [1.1, 1.6, 2.1])
flavinoid['alto'] = fuzz.trimf(flavinoid.universe, [2, 4, 6])

flavinoid.view()

tipo['classe_0'] = fuzz.trimf(tipo.universe, [0, 0, 1])
tipo['classe_1'] = fuzz.trimf(tipo.universe, [0.5, 1, 1.5])
tipo['classe_2'] = fuzz.trimf(tipo.universe, [1.5, 2, 2])

tipo.view()  #grafico fuzzy saida tipo

rule1 = ctrl.Rule(collor['rosa'] , tipo['classe_1'])
rule2 = ctrl.Rule((collor['vermelho_claro']| collor['vermelho_escuro']) &  flavinoid['baixo'] , tipo['classe_2'])
rule3 = ctrl.Rule((collor['vermelho_claro']| collor['vermelho_escuro']) & (flavinoid['médio'] | flavinoid['alto']) , tipo['classe_0'])

#comandos de simulçao
classe_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
classe = ctrl.ControlSystemSimulation(classe_ctrl)

# entrada de dados
wine = load_wine() 
Y=[]


for i in range(0,178,1):

    a = wine.data[i,[6]]
    b = wine.data[i,[9]]
    
    Flavinoide = float(np.asarray(a))  
    Coloração = float(np.asarray(b))
    
    classe.input['Flavinoides'] = Flavinoide
    classe.input['Coloração'] = Coloração

    classe.compute()
    if (classe.output['tipo']<0.8):
        Y.append(0)
    if (classe.output['tipo']<1.1):
            if (classe.output['tipo']>0.80):
                Y.append(1)
                
    if (classe.output['tipo']>1.1):
                    Y.append(2)
    
    
    
    x = wine.target

    
print(classe.output['tipo'])

tipo.view(sim=classe)

print(accuracy_score( Y, x )) 

print(classification_report(Y, x))




