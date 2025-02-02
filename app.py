import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import gradio as gr

# Debugging code to print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir())

# Load the model and encoder
try:
    with open("model.pkl", "rb") as model_file:
        gbc = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open("encoder.pkl", "rb") as encoder_file:
        encoder_dict = pickle.load(encoder_file)
    print("Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading encoder: {e}")

def predict_employability(Age, Accessibility, EdLevel, Employment, Gender, MentalHealth, MainBranch, YearsCode, PreviousSalary, ComputerSkills, Continent):
    data = {
        'Age': Age, 'Accessibility': Accessibility, 'EdLevel': EdLevel, 'Employment': Employment, 
        'Gender': Gender, 'MentalHealth': MentalHealth, 'MainBranch': MainBranch, 
        'YearsCode': YearsCode, 'PreviousSalary': PreviousSalary, 'ComputerSkills': ComputerSkills, 
        'Continent': Continent
    }
    print("Input data:", data)
    df = pd.DataFrame([list(data.values())], columns=[
        'Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender', 'MentalHealth', 
        'MainBranch', 'YearsCode', 'PreviousSalary', 'ComputerSkills', 'Continent'
    ])
    print("DataFrame before encoding:", df)
    category_col = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Continent']
    for cat in encoder_dict:
        for col in df.columns:
            le = preprocessing.LabelEncoder()
            if cat == col:
                le.classes_ = np.array(encoder_dict[cat], dtype=object)
                for unique_item in df[col].unique():
                    if unique_item not in le.classes_:
                        df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
                df[col] = le.transform(df[col].astype(str))
    print("DataFrame after encoding:", df)
    
    # Convert categorical values to numerical values
    df['Age'] = df['Age'].map({'<35': 0, '>35': 1})
    df['Accessibility'] = df['Accessibility'].map({'No': 0, 'Yes': 1})
    df['EdLevel'] = df['EdLevel'].map({'Master': 0, 'NoHigherEd': 1, 'Other': 2, 'PhD': 3, 'Undergraduate': 4})
    df['Gender'] = df['Gender'].map({'Man': 0, 'NonBinary': 1, 'Woman': 2})
    df['MentalHealth'] = df['MentalHealth'].map({'No': 0, 'Yes': 1})
    df['MainBranch'] = df['MainBranch'].map({'Dev': 0, 'NotDev': 1})
    df['Continent'] = df['Continent'].map({'Africa': 0, 'Asia': 1, 'Europe': 2, 'North_America': 3, 'Oceania': 4, 'Others': 5, 'South_America': 6})
    
    print("DataFrame after mapping:", df)
    features_list = df.values.tolist()
    print("Features list:", features_list)
    try:
        prediction = gbc.predict(features_list)
        print("Prediction:", prediction)
        return "Employable(Demonstrates Relevant Skills and Experience.)" if prediction[0] == 1 else "Less Employable(Profile Requires Slight Enhancement For Improved Employability.)"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

iface = gr.Interface(
    fn=predict_employability,
    inputs=[
        gr.Dropdown(['<35', '>35'], label="Age"),
        gr.Dropdown(['No', 'Yes'], label="Accessibility Requirements"),
        gr.Dropdown(['Master', 'NoHigherEd', 'Other', 'PhD', 'Undergraduate'], label="Level of Education"),
        gr.Slider(0, 1, step=1, label="Ever been Employed or Interned(0 for No/1 for Yes)"),
        gr.Dropdown(['Man', 'NonBinary', 'Woman'], label="Gender"),
        gr.Dropdown(['No', 'Yes'], label="Experienced Mental Health Challenges"),
        gr.Dropdown(['Dev', 'NotDev'], label="Your Branch of Work"),
        gr.Slider(0, 50, step=1, label="Years of Experience"),
        gr.Number(label="Previous Salary"),
        gr.Slider(0, 100, step=1, label="Number of Computer Skills You Posess"),
        gr.Dropdown(['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'Others', 'South_America'], label="The Continent You Belong to")
    ],
    outputs="text",
    title="Employability Prediction",
    description="Predict employability based on various features."
)

iface.launch()
