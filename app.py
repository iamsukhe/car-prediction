from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
model = pickle.load(open("car_price_prediction_model.pkl", "rb"))
car = pd.read_csv("cleaned_car_data.csv")


@app.route("/")
def index():
    # Get unique values for dropdowns
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_types = car["fuel_type"].unique()
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_types,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Extract data from form
    company = request.form.get("company")
    car_model = request.form.get("car_model")
    year = int(request.form.get("year"))
    fuel_type = request.form.get("fuel_type")
    kms_driven = int(request.form.get("kilo_driven"))

    # Your model expects a DataFrame as input
    input_data = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )

    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))  # Return result to frontend


if __name__ == "__main__":
    app.run(debug=True)
