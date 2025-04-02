import streamlit as st
import joblib
import pandas as pd
import numpy as np 
import math

# # # Load the saved pipeline and trained model
# preprocessor = joblib.load('preprocessing_pipeline.pkl')
# price_scaler = joblib.load('price_scaler.pkl')
# model = joblib.load('prediction_model')



# # Streamlit App
# st.title("📱 Mobile Price Predictor")
# st.write("Enter the mobile specifications below to predict its price.")

# # User Inputs

# df= pd.read_csv('C:\\Users\\HP\Desktop\\ML models\\myenv\mobile_data.xls')


# processor = st.selectbox("Processor", df['Processor'].unique())

# brand = st.selectbox("Brand",df['Brand'].unique())

# rating = st.slider("Rating (out of 5)", 1.0, 5.0, 4.5)
# battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4500)
# is_5g = st.radio("Is it 5G?", [0, 1])
# ram = st.selectbox("RAM (GB)", [4, 6, 8, 12, 16])
# rom = st.selectbox("Storage (GB)", [64, 128, 256, 512])
# rear_camera = st.number_input("Rear Camera (MP)", min_value=8, max_value=200, value=64)
# front_camera = st.number_input("Front Camera (MP)", min_value=5, max_value=50, value=32)
# display = st.number_input("Screen Size (inches)", min_value=4.0, max_value=7.5, value=6.7)
# is_amoled = st.radio("Is it AMOLED Display?", [0, 1])

# # Predict Button
# if st.button("Predict Price"):
#     # Create input DataFrame
#     input_data = pd.DataFrame({'Rating': [rating],
#                   'Battery': [battery],
#                    'Is5G': [is_5g],
#                    'RAM': [ram],
#                    'ROM': [rom],
#                    'rear_camera': [rear_camera],
#                    'front_camera': [front_camera],
#                    'display': [display],
#                    'IsAMOLED': [is_amoled],
#                    'Processor': [processor],
#                    'Brand': [brand],})

#     # Transform input using the saved preprocessing pipeline
#     processed_input = preprocessor.transform(input_data)
#     print( processed_input)
#     # Make a scaled prediction
#     predicted_price_scaled = model.predict(processed_input)

#     # Convert back to original price range
#     predicted_price = price_scaler.inverse_transform(np.array(predicted_price_scaled).reshape(-1, 1))

#     # Display the result
#     st.success(f"💰 Predicted Price: ₹{predicted_price[0][0]:,.2f}")

# import os
# import streamlit as st

  
# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# Load the saved pipeline and trained model
preprocessor = joblib.load('preprocessing_pipeline.pkl')
price_scaler = joblib.load('price_scaler.pkl')
model = joblib.load('prediction_model')


# Streamlit App
st.title("📱 Mobile Price Predictor")
st.write("Enter the mobile specifications below to predict its price.")

# Load Dataset

df = pd.read_csv('mobile_data.xls')

# Debug: Show some sample data


# User Inputs
processor = st.selectbox("Processor", df['Processor'].unique())
brand = st.selectbox("Brand", df['Brand'].unique())

rating = st.slider("Rating (out of 5)", 1.0, 5.0, 4.5)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4500)
is_5g = st.radio("Is it 5G?", [0, 1])
ram = st.selectbox("RAM (GB)", [4, 6, 8, 12, 16])
rom = st.selectbox("Storage (GB)", [64, 128, 256, 512])
rear_camera = st.number_input("Rear Camera (MP)", min_value=8, max_value=200, value=64)
front_camera = st.number_input("Front Camera (MP)", min_value=5, max_value=50, value=32)
display = st.number_input("Screen Size (inches)", min_value=4.0, max_value=7.5, value=6.7)
is_amoled = st.radio("Is it AMOLED Display?", [0, 1])

# Predict Button
if st.button("Predict Price"):
    st.write("Processing input data...")
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Rating': [rating],
        'Battery': [battery],
        'Is5G': [is_5g],
        'RAM': [ram],
        'ROM': [rom],
        'rear_camera': [rear_camera],
        'front_camera': [front_camera],
        'display': [display],
        'IsAMOLED': [is_amoled],
        'Processor': [processor],
        'Brand': [brand],
    })

    # Transform input using the saved preprocessing pipeline
    try:
        processed_input = preprocessor.transform(input_data)
        

        # Make a scaled prediction
        predicted_price_scaled = model.predict(processed_input)

        # Convert back to original price range
        predicted_price = price_scaler.inverse_transform(np.array(predicted_price_scaled).reshape(-1, 1))

        # Display the result
        st.success(f"💰 Predicted Price: ₹{predicted_price[0][0]:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


