from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# 1️⃣ Load your trained MultiOutputRegressor model
model = joblib.load('multioutput_model.pkl')  # must be in the same folder as app.py

# 2️⃣ Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')  # templates/index.html

# 3️⃣ Predict route to handle uploaded CSV
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # 4️⃣ Read uploaded CSV
    df_test = pd.read_csv(file)

    # 5️⃣ Make predictions
    # Drop 'ID' column if present for prediction
    features = df_test.drop(columns=['ID'], errors='ignore')
    predictions = model.predict(features)

    # 6️⃣ Prepare output DataFrame
    output_df = pd.DataFrame(predictions, columns=[
        'BlendProperty1','BlendProperty2','BlendProperty3','BlendProperty4','BlendProperty5',
        'BlendProperty6','BlendProperty7','BlendProperty8','BlendProperty9','BlendProperty10'
    ])

    # Include ID column if present
    if 'ID' in df_test.columns:
        output_df.insert(0, 'ID', df_test['ID'])

    # 7️⃣ Return as HTML table
    return output_df.to_html(classes='table table-striped', index=False)

# 8️⃣ Run the app
if __name__ == '__main__':
    app.run(debug=True)
