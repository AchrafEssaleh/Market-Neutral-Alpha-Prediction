import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DATA_PATH = "./data"
OUTPUT_FILE = "submission_baseline.csv"

# --- 2. CHARGEMENT DES DONNÉES ---
print("Chargement des données...")
x_train = pd.read_csv(f'{DATA_PATH}/X_train.csv', index_col='ID')
y_train = pd.read_csv(f'{DATA_PATH}/y_train.csv', index_col='ID')
test = pd.read_csv(f'{DATA_PATH}/X_test.csv', index_col='ID')

train = pd.concat([x_train, y_train], axis=1)
print(f"Données chargées. Train: {train.shape}, Test: {test.shape}")