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
