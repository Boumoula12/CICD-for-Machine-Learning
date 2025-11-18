import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1) Charger le dataset Iris
# Assure-toi que ton fichier s'appelle bien Data/iris.csv
df = pd.read_csv("Data/iris.csv")

# Adapter ces noms de colonnes à ton fichier si besoin
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_col = "species"

X = df[feature_cols]
y = df[target_col]

# 2) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3) Pipeline : StandardScaler + RandomForest
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ]
)

pipe.fit(X_train, y_train)

# 4) Évaluation
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}")

# 5) Créer les dossiers si besoin
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# 6) Sauvegarder les métriques dans Results/metrics.txt
with open("Results/metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy = {acc:.3f}, F1 Score = {f1:.3f}\n")

# 7) Sauvegarder la matrice de confusion dans Results/model_results.png
cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

# 8) Sauvegarder le modèle complet dans Model/model.pkl
joblib.dump(pipe, "Model/model.pkl")

print("✅ Entraînement terminé, modèle et résultats sauvegardés.")
