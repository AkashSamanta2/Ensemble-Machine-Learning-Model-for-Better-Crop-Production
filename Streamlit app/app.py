import streamlit as st
import pandas as pd
import pickle
import numpy as np


file1 = open('Pickle_ensemble_models/Ensemble_AdaBayes.pkl', "rb")
AdaBayes = pickle.load(file1)
file2 = open('Pickle_ensemble_models/Ensemble_AdaForest.pkl', "rb")
AdaForest = pickle.load(file2)
file3 = open('Pickle_ensemble_models/Ensemble_NaiveForest.pkl', "rb")
NaiveForest = pickle.load(file3)


# Streamlit app
st.title('Crop Recommendation System')

st.sidebar.image('logobg.png', width=290)
menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":

    choice_model = st.sidebar.radio('Choose Model', ["AdaBayes", "AdaForest", "NaiveForest"])

    if choice_model == "AdaBayes":
        # Input features
        st.subheader('Input Features')

        col1, col2 = st.columns(2)
        
        with col1:
            

            N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=250.0, value=25.0)
            P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=250.0, value=25.0)
            K = st.number_input('Potassium (K)', min_value=0.0, max_value=250.0, value=25.0)
            pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)

        with col2:

            temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=70.0, value=30.0)
            humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=150.0, value=50.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=350.0, value=100.0)


        # Prepare the feature vector for prediction
        features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Predict and display the result
        if st.button('Predict Crop'):

            prediction = AdaBayes.predict(features)

            st.success("AdaBayes Result:")
            st.write(f'The predicted crop is: {prediction}')
    
    if choice_model == "AdaForest":

        # Input features
        st.subheader('Input Features')

        col1, col2 = st.columns(2)
        
        with col1:
            

            N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=250.0, value=25.0)
            P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=250.0, value=25.0)
            K = st.number_input('Potassium (K)', min_value=0.0, max_value=250.0, value=25.0)
            pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)

        with col2:

            temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=70.0, value=30.0)
            humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=150.0, value=50.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=350.0, value=100.0)

        # Prepare the feature vector for prediction
        features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Predict and display the result
        if st.button('Predict Crop'):

            prediction = AdaForest.predict(features)

            st.success("AdaForest Result:")
            st.write(f'The predicted crop is: {prediction}')
    
    if choice_model == "NaiveForest":

        # Input features
        st.subheader('Input Features')

        col1, col2 = st.columns(2)
        
        with col1:
            

            N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=250.0, value=25.0)
            P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=250.0, value=25.0)
            K = st.number_input('Potassium (K)', min_value=0.0, max_value=250.0, value=25.0)
            pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)

        with col2:

            temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=70.0, value=30.0)
            humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=150.0, value=50.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=350.0, value=100.0)

        # Prepare the feature vector for prediction
        features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        
        # Predict and display the result
        if st.button('Predict Crop'):

            prediction = NaiveForest.predict(features)

            st.success("NaiveForest Result:")
            st.write(f'The predicted crop is: {prediction}')

if choice == "About":
    st.header("About the Crop Recommendation System")
    
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("logo.png", use_column_width=True)
    

    st.subheader("Overview")
    st.markdown("""
    The Crop Recommendation System is a powerful tool designed to assist farmers and agricultural professionals in making informed decisions about crop selection based on soil and weather conditions. Leveraging advanced machine learning techniques, this system provides accurate and reliable crop recommendations to optimize yield and ensure sustainable farming practices.
    """)

    st.subheader("How It Works")
    st.markdown("""
    Our system utilizes ensemble machine learning models to predict the best crop to plant. The following models are available in the system:
    - **AdaBayes**: Combines Adaptive Boosting (AdaBoost) with Naive Bayes.
    - **AdaForest**: Merges the strengths of Adaptive Boosting with Random Forest.
    - **NaiveForest**: Integrates Naive Bayes with Random Forest.
    
    These models analyze various soil and weather parameters to make predictions, including:
    - **Nitrogen (N)**: Essential nutrient for plant growth.
    - **Phosphorus (P)**: Important for root development and energy transfer.
    - **Potassium (K)**: Crucial for water regulation and enzyme activation.
    - **pH**: Soil acidity or alkalinity, affecting nutrient availability.
    - **Temperature (°C)**: Influences crop growth cycles.
    - **Humidity (%)**: Affects transpiration rates and disease prevalence.
    - **Rainfall (mm)**: Determines water availability for crops.
    """)

    st.subheader("Technologies Used")
    st.markdown("""
    The system is built using the following technologies:
    - **Streamlit**: For building the web interface.
    - **scikit-learn**: For implementing machine learning models.
    - **Pickle**: For efficient serialization of machine learning models.
    - **NumPy**: For numerical computations.
    - **Pandas**: For data manipulation and analysis.
    """)

    st.subheader("Model Evaluation Metrics")
    st.markdown("""
    To ensure the reliability of the crop recommendations, we evaluate our models using various metrics. Here are the performance metrics for the ensemble models used in this system:
    """)

    # Displaying a table with evaluation metrics
    eval_data = {
        "Model": ["AdaBayes", "AdaForest", "NaiveForest"],
        "Accuracy": [0.92, 0.94, 0.91],
        "Precision": [0.93, 0.95, 0.92],
        "Recall": [0.91, 0.93, 0.90],
        "F1-Score": [0.92, 0.94, 0.91]
    }

    eval_df = pd.DataFrame(eval_data)
    eval_df.index = np.arange(1, len(eval_df) + 1)
    st.table(eval_df)

    st.subheader("Usage Instructions")
    st.markdown("""
    1. Navigate to the 'Home' section from the sidebar.
    2. Select the machine learning model you wish to use (AdaBayes, AdaForest, or NaiveForest).
    3. Enter the required soil and weather parameters in the input fields.
    4. Click on the 'Predict Crop' button to get the recommended crop.
    """)

    st.subheader("About the Developer")
    st.markdown("""
    Developed by, a group of passionate students of Techno International Batanagar with expertise in machine learning and a keen interest in agricultural technology. For inquiries or collaborations, please reach out us via our email or connect on LikedIn.
    1. Arya Bose, aryabose2001@gmail.com, [Arya_Bose_LinkedIn](https://www.linkedin.com/in/arya-bose-655a77245)
    2. Aditya Ghosh, adityaghoshfkt00@gmail.com, [Aditya_Ghosh_LinkedIn](https://www.linkedin.com/in/aditya-ghosh-52237a2b1)
    3. Akash Samanta, akashsamanta963@gmail.com, [Akash_Samanta_LinkedIn](https://www.linkedin.com/in/akash-samanta-727647211)
    4. Aditya Ray, bobbyrayb98@gmail.com, [Aditya_Ray_LinkedIn](https://www.linkedin.com/in/aditya-ray-7772b7215)
    """)

    st.subheader("Acknowledgements")
    st.markdown("""
    We would like to extend our sincere thanks to the following individuals and organizations for their support and contributions to this project:
    - **Prof.(Dr.) Ratikanto Sahoo**, Director, Techno International Batanagar, for his visionary leadership and support.
    - **Mr. Subhankar Guha**, HOD, CSE Department, for his guidance and encouragement throughout the development of this project.
    - **Ms. Tanushree Chakraborty**, our Project Guide, for her invaluable insights, feedback, and continuous support.
    - Kaggle for providing datasets and a collaborative platform for data science.
    - The open-source community for continuous contributions to the tools and libraries used in this project.
    """)


