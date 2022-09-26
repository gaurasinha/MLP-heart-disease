# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:54:31 2022

@author: Gaura
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics

heart = pd.read_csv('heart.csv')
#print(heart)
#np.random.shuffle(heart.DataValues)
y = heart['target']
X = heart[['oldpeak','thal']]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=2000,activation = 'relu',solver='adam',random_state=1)

sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

classifier.fit(X_trainscaled, y_train)
y_pred = classifier.predict(X_testscaled)
print(y_pred)
print("\n\nAccuracy= ", metrics.accuracy_score(y_test,y_pred))
print("Recall= ", metrics.recall_score(y_test,y_pred))
print("Precision= ", metrics.precision_score(y_test,y_pred))
print("F1= ", metrics.f1_score(y_test,y_pred))
