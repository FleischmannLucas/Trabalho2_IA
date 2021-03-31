# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:50:47 2021

@author: Gustavo
"""

from sklearn import datasets
import pandas as pd
import seaborn as sns

iris=datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['species']= pd.Categorical.from_codes(iris.target, iris.target_names)

iris_df.head()

sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)','species']], hue='species')
