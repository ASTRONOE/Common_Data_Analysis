import streamlit as st
import diamonds as dd

# create an instance of DiamondsPredictor
dd.load_model('diamond_price_rf_reg.joblib')

# set the page title and description
st.title("Diamonds Price Predictor")
st.write("This app predicts the price of a diamond based on its characteristics.")

# create dropdowns for cut, color, and clarity
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

col1, col2, col3 = st.columns(3)
cut = col1.selectbox('Cut', cut_options)
color = col2.selectbox('Color', color_options)
clarity = col3.selectbox('Clarity', clarity_options)


# create a slider for carat and display the predicted price
carat = st.slider('Amount of carats', 0.0, 5.0, 1.0, 0.1)
diam = {'cut': cut, 'color': color, 'clarity': clarity, 'carat': carat}

predicted_price_rf = dd.rf_predict_price(diam)

st.write('The price of this diamond is', round(predicted_price_rf, 2))
