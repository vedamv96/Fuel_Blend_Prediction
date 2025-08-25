from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("multioutput_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Check for uploaded file
    if 'file' not in request.files:
        return render_template("index.html", tables=None)
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", tables=None)

    # Read CSV
    df_test = pd.read_csv(file)

    # Prepare features
    features = df_test.drop(columns=['ID'], errors='ignore')

    # Make predictions
    predictions = model.predict(features)

    # Prepare output DataFrame
    output_df = pd.DataFrame(predictions, columns=[
        'BlendProperty1','BlendProperty2','BlendProperty3','BlendProperty4','BlendProperty5',
        'BlendProperty6','BlendProperty7','BlendProperty8','BlendProperty9','BlendProperty10'
    ])

    # Include ID column if present
    if 'ID' in df_test.columns:
        output_df.insert(0, 'ID', df_test['ID'])

    # Convert DataFrame to HTML table
    table_html = output_df.to_html(classes='table table-striped', index=False)

    return render_template("index.html", tables=table_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
