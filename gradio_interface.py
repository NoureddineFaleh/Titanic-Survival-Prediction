import gradio as gr
import numpy as np
import pickle

# Charger le modèle de régression logistique
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression = pickle.load(file)

def predict_survival(sex, pclass, fare):
    # Convertir les entrées en format approprié
    sex = 0 if sex == "Male" else 1
    user_data = np.array([[sex, fare, pclass]])
    
    # Utiliser le modèle de régression logistique pour faire une prédiction
    survival_prediction = logistic_regression.predict(user_data)
    
    # Afficher le résultat de la prédiction
    if survival_prediction[0] == 1:
        return "The person would have survived."
    else:
        return "The person would not have survived."

# Définir les composants de l'interface
sex_input = gr.Radio(choices=["Male", "Female"], label="Gender")
pclass_input = gr.Dropdown(choices=[1, 2, 3], label="Class")
fare_input = gr.Number(label="Fare")

# Créer l'interface Gradio
interface = gr.Interface(fn=predict_survival,
                         inputs=[sex_input, pclass_input, fare_input],
                         outputs="text",
                         title="Titanic Survival Prediction",
                         description="Predict whether a person would have survived the Titanic disaster based on their gender, class, and fare paid.",
                         article="Created by Noureddine Faleh. All rights reserved.")

# Lancer l'interface
interface.launch()
