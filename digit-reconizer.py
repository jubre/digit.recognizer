# Analisis de datos y Manejo
import pandas as pd
import numpy as np
import random as rnd

# Visualizacion
import seaborn as sns
import matplotlib.pyplot as plt

# Lectura de los datos
train_df = pd.read_csv('~/.kaggle/competitions/digit-recognizer/train.csv')
test_df = pd.read_csv('~/.kaggle/competitions/digit-recognizer/test.csv')
combine = [train_df, test_df]
