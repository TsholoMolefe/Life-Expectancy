import streamlit as st
import pickle
import numpy as np

# Load the trained regression model
with open('C:\\Users\\Administrator\\Desktop\\App\model\\regression.pkl', 'rb') as file:
    
    regression_model = pickle.load(file)

# Function to predict using the loaded model
def predict(model, input_data):
    # Assuming your model.predict function takes a 2D array as input
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    # Streamlit app header
    st.title('Life Expectancy Predictor')

    # Input features from user using dropdowns
    year = st.selectbox('Select a year', list(range(2000, 2031)), index=20)  # Default to 2020
    adult_mortality = st.selectbox('Adult Mortality', list(range(701)), index=50)  # Default to 50
    under_five_deaths = st.selectbox('Under Five Deaths', list(range(201)), index=20)  # Default to 20
    diphtheria = st.selectbox('Diphtheria', list(range(101)), index=20)  # Default to 20
    HIV = st.selectbox('HIV', list(range(101)), index=20)  # Default to 20
    GDP = st.selectbox('GDP', list(range(20001)), index=1)  # Default to 1
    Income_composition_of_resources = st.selectbox('Income composition of resources', np.arange(0.0, 3.6, 0.1), index=11)  # Default to 1.1

    # Create a feature vector from user inputs
    input_features = [year, adult_mortality, under_five_deaths, diphtheria, HIV, GDP, Income_composition_of_resources]

    # Make a prediction
    prediction = predict(regression_model, input_features)

    # Display the prediction
    st.subheader('Prediction:')
    st.write(f'The predicted value is: {prediction}')

if __name__ == '__main__':
    main()
