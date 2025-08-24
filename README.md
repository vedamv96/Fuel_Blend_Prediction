Fuel Blend Properties Prediction
Project Overview

Blending fuels to achieve desired properties is a high-dimensional and complex challenge. Each fuel blend is composed of multiple components with unique chemical and physical characteristics. The final blend properties depend on non-linear interactions and synergistic effects between components.

This project develops a machine learning solution that predicts 10 critical fuel blend properties based on:

Blend composition – volume fractions of base components.

Component properties – real-world COA (Certificate of Analysis) data for each batch.

Objective

Enable rapid evaluation of thousands of potential fuel blend combinations.

Identify optimal recipes balancing performance, safety, sustainability, and cost.

Reduce development time for new sustainable fuel formulations.

Support real-time blend optimization in production facilities.

Accurate predictions accelerate the adoption of Sustainable Aviation Fuels (SAFs), helping the global aviation industry transition to a net-zero future.

Project Features

Predicts 10 final blend properties from input component data.

Uses MultiOutputRegressor to handle multiple outputs simultaneously.

Web-based interface via Flask, allowing CSV uploads for batch predictions.

Professional HTML front-end for easy interaction.

Outputs prediction tables in real-time, ready for evaluation and optimization.

Folder Structure
fuel_blend/
├─ model.ipynb                 # Model training notebook
├─ train.csv                   # Training dataset
├─ test.csv                    # Test dataset for predictions
├─ multioutput_model.pkl       # Saved trained model
├─ app.py                      # Flask web application
└─ templates/
    └─ index.html              # HTML upload form

Technologies Used

Python, Pandas, Scikit-learn, XGBoost

Joblib (for model serialization)

Flask (for web interface)

HTML/CSS (professional UI)
