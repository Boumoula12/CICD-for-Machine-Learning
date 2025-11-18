import gradio as gr
import joblib
import pandas as pd

# Charger le modèle entraîné
model = joblib.load("Model/model.pkl")

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Construire un DataFrame avec une seule ligne
    data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    prediction = model.predict(data)[0]
    return f"Classe prédite : {prediction}"

inputs = [
    gr.Slider(4.0, 8.0, step=0.1, label="Sepal length"),
    gr.Slider(2.0, 4.5, step=0.1, label="Sepal width"),
    gr.Slider(1.0, 7.0, step=0.1, label="Petal length"),
    gr.Slider(0.1, 2.5, step=0.1, label="Petal width"),
]
output = gr.Label()

title = "Iris Classification"
description = "Application de démonstration pour le CI/CD : classification des fleurs Iris."

demo = gr.Interface(
    fn=predict_iris,
    inputs=inputs,
    outputs=output,
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()
