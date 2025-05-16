from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = None
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        # New input features
        doornumber = request.form['doornumber']
        carbody = request.form['carbody']
        wheelbase = float(request.form['wheelbase'])
        carlength = float(request.form['carlength'])
        carwidth = float(request.form['carwidth'])
        carheight = float(request.form['carheight'])
        curbweight = float(request.form['curbweight'])
        enginetype = request.form['enginetype']
        enginesize = int(request.form['enginesize'])
        stroke = float(request.form['stroke'])
        horsepower = int(request.form['horsepower'])
        peakrpm = int(request.form['peakrpm'])
        citympg = int(request.form['citympg'])
        highwaympg = int(request.form['highwaympg'])

        # Calculate Car_Age
        from datetime import datetime
        current_year = datetime.now().year
        car_age = current_year - year

        # Prepare input array in the order expected by the model
        import pandas as pd
        input_dict = {
            'fueltype': [fuel_type],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'doornumber': [doornumber],
            'carbody': [carbody],
            'wheelbase': [wheelbase],
            'carlength': [carlength],
            'carwidth': [carwidth],
            'carheight': [carheight],
            'curbweight': [curbweight],
            'enginetype': [enginetype],
            'enginesize': [enginesize],
            'stroke': [stroke],
            'horsepower': [horsepower],
            'peakrpm': [peakrpm],
            'citympg': [citympg],
            'highwaympg': [highwaympg],
            'Owner': [owner],
            'Car_Age': [car_age],
            'Kms_Driven': [kms_driven],
            'Present_Price': [present_price]
        }
        input_df = pd.DataFrame(input_dict)

        # Predict price
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f'Estimated Car Price: ₹ {prediction} Lakhs')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
