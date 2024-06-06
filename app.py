import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
model = load_model('lstm_stock_prediction_model.h5')

# Tokenizer settings
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='UNK')

# Hardcoded username and password
USERNAME = "admin"
PASSWORD = "123"

# Initialize Hugging Face model
tokenizer_hf = GPT2Tokenizer.from_pretrained("gpt2")
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate a response
def generate_response(user_input):
    input_ids = tokenizer_hf.encode(user_input + tokenizer_hf.eos_token, return_tensors='pt')
    chat_history_ids = model_hf.generate(input_ids, max_length=1000, pad_token_id=tokenizer_hf.eos_token_id)
    response = tokenizer_hf.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Function to display the login page
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password")

# Function to display the home page
def home():
    st.title("Home Page")
    tabs = st.tabs(["Prediction", "Chatbot"])
    
    with tabs[0]:
        st.header("Stock Prediction")
        # Input fields
        product_name = st.text_input('Product Name/Type')
        sales_per_week = st.number_input('Sales Per Week/Month', min_value=0)
        product_lifetime = st.number_input('Product Lifetime (in days)', min_value=0)
        transaction_city = st.text_input('Transaction City')
        transaction_state = st.text_input('Transaction State')
        analysis_month = st.number_input('Analysis Month', min_value=1, max_value=12)

        if st.button('Predict'):
            # Combine features for prediction
            combined_features = f"{product_name} {transaction_state}"

            # Tokenize the input
            tokenizer.fit_on_texts([combined_features])
            sequences = tokenizer.texts_to_sequences([combined_features])
            input_data = pad_sequences(sequences, maxlen=100)

            # Make prediction
            prediction = model.predict(input_data)
            st.write(f"Predicted Stock Availability: {'In Stock' if prediction[0][0] > 0.5 else 'Out of Stock'}")

            # Display additional analysis
            st.write(f"Analysis Month: {analysis_month}")
            st.write(f"Sales Per Week/Month: {sales_per_week}")
            st.write(f"Product Lifetime: {product_lifetime} days")
            st.write(f"Transaction Location: {transaction_city}, {transaction_state}")

    with tabs[1]:
        st.header("Customer Assistance Chatbot")
        # Chatbot interaction
        user_input = st.text_input("You: ", "")
        if user_input:
            response = generate_response(user_input)
            st.text_area("Chatbot:", value=response, height=200, max_chars=None, key=None)

# Main function to control the app flow
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        home()
    else:
        login()

# Run the main function
if __name__ == "__main__":
    main()
