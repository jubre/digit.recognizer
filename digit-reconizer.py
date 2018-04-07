# Analisis de datos y Manejo
import pandas as pd
import numpy as np
import random as rnd

# Visualizacion
import seaborn as sns
import matplotlib.pyplot as plt

# Lectura de los datos
train = pd.read_csv('~/.kaggle/competitions/digit-recognizer/train.csv')
test = pd.read_csv('~/.kaggle/competitions/digit-recognizer/test.csv')
combine = [train, test]

# Describiendo la data
print(train.columns.values)

# Previsualizacion de la data de entrenamiento
print(train.head())
print(train.tail())

# Obteniendo la columna de 'label' de la data de entrenamiento
Y_train = train["label"]

# Eliminando la columna de 'label' de la data de entrenamiento
X_train = train.drop(labels = ["label"],axis = 1)

print('='*80)
print(X_train.head())
print(X_train.tail())

# Eliminamos la matriz de entreamiento original para liberar espacio
del train
g = sns.countplot(Y_train)
print(Y_train.value_counts())
