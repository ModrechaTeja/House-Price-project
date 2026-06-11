from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["OverallQual"]),
        float(request.form["GrLivArea"]),
        float(request.form["GarageCars"]),
        float(request.form["TotalBsmtSF"]),
        float(request.form["YearBuilt"])
    ]

    prediction = model.predict([features])[0]

    return render_template(
        "index.html",
         prediction_text=f"Predicted House Price: ${prediction:,.0f}"
    )

if __name__ == "__main__":
    app.run(debug=True)