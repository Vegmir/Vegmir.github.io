# Tarea 2 – Clasificador binario con Random Forest y KNN

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Cargar el dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. Convertir etiquetas a formato binario (¿Es 7?)
y_binario = (y == 7).astype(int)

# 3. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binario, test_size=0.2, random_state=42)

# 4. Entrenar modelos
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=42)

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 5. Evaluación
def evaluar_modelo(nombre, modelo):
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"--- {nombre} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    return y_pred, modelo.predict_proba(X_test)[:, 1]

y_pred_knn, scores_knn = evaluar_modelo("KNN", knn)
y_pred_rf, scores_rf = evaluar_modelo("Random Forest", rf)

# 6. Seleccionar mejor modelo
mejor_modelo = rf if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_knn) else knn
scores_mejor = scores_rf if mejor_modelo == rf else scores_knn
y_pred_mejor = y_pred_rf if mejor_modelo == rf else y_pred_knn

# 7. Visualizaciones
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_mejor)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# Curva Precisión vs Recall
precisions, recalls, thresholds = precision_recall_curve(y_test, scores_mejor)
plt.figure()
plt.plot(recalls, precisions, color="b")
plt.title("Precisión vs Recall")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.grid(True)
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, scores_mejor)
roc_auc = roc_auc_score(y_test, scores_mejor)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()







# --- Importar librerías ---
import pandas as pd
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile

# --- Ver archivos en el directorio actual ---
print("Archivos disponibles:")
print(os.listdir())

# --- Función utilitaria para extraer datos ---
def unzip_data(path):
    with ZipFile(path, 'r') as zipObj:
        zipObj.extractall()
    print(f"Datos extraídos desde {path}")

# --- Extraer archivo ZIP ---
unzip_data('spaceship-titanic.zip')

# --- Leer datasets extraídos ---
train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')

# --- Visualizar las primeras filas ---
print("\nPrimeras 5 filas del dataset de entrenamiento:")
print(train_ds.head())

print("\nPrimeras 5 filas del dataset de prueba:")
print(test_ds.head())

# --- Cantidad de filas ---
ntrain = train_ds.shape[0]
ntest = test_ds.shape[0]
print(f'\nDataset tiene {ntrain} datos de entrenamiento')
print(f'Dataset tiene {ntest} datos de prueba')

# --- Información de las columnas ---
print("\nInformación general del dataset de entrenamiento:")
print(train_ds.info())

# --- Verificar campos nulos ---
print("\nCampos nulos en el dataset de entrenamiento:")
print(train_ds.isnull().sum())

# --- Función para imputar valores más frecuentes ---
def impute_most_frequent_data(df):
    for column_name in df.columns:
        if df[column_name].isnull().sum() > 0:
            most_frequent = df[column_name].value_counts().idxmax()
            df[column_name].fillna(most_frequent, inplace=True)
    return df

# --- Aplicar la imputación ---
train_ds = impute_most_frequent_data(train_ds)

# --- Validar campos nulos nuevamente ---
print("\nCampos nulos después de imputación:")
print(train_ds.isnull().sum())

# --- Agrupar por planeta y sumar la columna VIP ---
home_planet_vs_vip = train_ds.groupby('HomePlanet')['VIP'].sum()
print("\nCantidad de VIP por planeta de origen:")
print(home_planet_vs_vip)

# --- Gráfico de barras de VIPs por planeta ---
fig, ax = plt.subplots()
ax.bar(home_planet_vs_vip.index, home_planet_vs_vip.values)
ax.set_xticklabels(home_planet_vs_vip.index, rotation=45)
ax.set_ylabel("Cantidad de personas VIP por planeta")
ax.set_title("VIPs por HomePlanet")
plt.tight_layout()
plt.show()

# --- Agrupar por edad y sumar gastos totales ---
train_ds['TotalSpent'] = train_ds[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
age_vs_moneyspent = train_ds.groupby('Age')['TotalSpent'].sum()
print("\nGasto total por edad:")
print(age_vs_moneyspent.head())

# --- Gráfico de dispersión ---
plt.figure(figsize=(10, 5))
plt.scatter(age_vs_moneyspent.index, age_vs_moneyspent.values, color='purple', alpha=0.6)
plt.title("Dinero gastado por rango de edad")
plt.xlabel("Edad")
plt.ylabel("Gasto Total")
plt.grid(True)
plt.show()
