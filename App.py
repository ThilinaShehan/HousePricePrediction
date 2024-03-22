import streamlit as st
import numpy as np
import pickle

# Load the LGBMRegressor model from the pickle file
try:
    with open('house_price_prediction.pkl', 'rb') as f:
        model_lgb = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure the model file exists and is named 'house_price_prediction.pkl'.")
    st.stop()

# Function to preprocess input data
def preprocess_input(input_data):
    return np.array(input_data).reshape(1, -1)

# Function to make predictions
def predict(input_data):
    input_processed = preprocess_input(input_data)
    prediction = model_lgb.predict(input_processed)
    return prediction

# Streamlit app
def main():
    st.title('House Price Prediction')
    st.write('Enter the features to predict the house price:')
    
    # Input fields
    overall_qual = st.slider('OverallQual', 1, 10, 5)
    grlivarea = st.number_input('GrLivArea', value=1500)
    garage_cars = st.number_input('GarageCars', min_value=0, value=1)
    garage_area = st.number_input('GarageArea', value=0)
    total_bsmt_sf = st.number_input('TotalBsmtSF', value=0)
    first_flr_sf = st.number_input('1stFlrSF', value=0)
    full_bath = st.number_input('FullBath', min_value=0, value=1)
    tot_rms_abv_grd = st.number_input('TotRmsAbvGrd', min_value=1, value=5)
    year_built = st.number_input('YearBuilt', min_value=1800, max_value=2022, value=2000)
    year_remod_add = st.number_input('YearRemodAdd', min_value=1800, max_value=2022, value=2000)
    mas_vnr_area = st.number_input('MasVnrArea', value=0)
    garage_yr_blt = st.number_input('GarageYrBlt', min_value=1800, max_value=2022, value=2000)
    fireplaces = st.number_input('Fireplaces', min_value=0, value=0)
    bsmt_fin_sf1 = st.number_input('BsmtFinSF1', value=0)
    
    # Predict button
    if st.button('Predict'):
        input_data = [overall_qual, grlivarea, garage_cars, garage_area, total_bsmt_sf, first_flr_sf,
                      full_bath, tot_rms_abv_grd, year_built, year_remod_add, mas_vnr_area, 
                      garage_yr_blt, fireplaces, bsmt_fin_sf1]
        try:
            prediction = predict(input_data)
            prediction_in_dollars = prediction[0] * 66190  # Multiply the prediction by 66,190
            st.success(f'The predicted house price is ${prediction_in_dollars:,.2f}')
        except Exception as e:
            st.error(f'An error occurred during prediction: {e}')

if __name__ == "__main__":
    main()
