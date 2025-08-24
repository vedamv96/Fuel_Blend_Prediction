# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Load your trained MultiOutputRegressor model
model = joblib.load('multioutput_model.pkl')  # ensure this file is in the same folder

# 2️⃣ App title
st.title("Fuel Blend Property Prediction")
st.markdown("Upload a CSV file with test data and get predicted Blend Properties.")

# 3️⃣ File uploader
uploaded_file = st.file_uploader("Upload your test.csv file", type="csv")

if uploaded_file:
    # 4️⃣ Read CSV
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df_test)

    # 5️⃣ Prepare features
    features = df_test.drop(columns=['ID'], errors='ignore')

    # 6️⃣ Make predictions
    predictions = model.predict(features)

    # 7️⃣ Prepare output DataFrame
    output_df = pd.DataFrame(predictions, columns=[
        'BlendProperty1','BlendProperty2','BlendProperty3','BlendProperty4','BlendProperty5',
        'BlendProperty6','BlendProperty7','BlendProperty8','BlendProperty9','BlendProperty10'
    ])

    # Include ID column if present
    if 'ID' in df_test.columns:
        output_df.insert(0, 'ID', df_test['ID'])

    # 8️⃣ Display predictions
    st.subheader("Predicted Blend Properties")
    st.dataframe(output_df)

    # 9️⃣ Allow user to download predictions
    csv = output_df.to_csv(index=False)
    st.download_button("Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')
