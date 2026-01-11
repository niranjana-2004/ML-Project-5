from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved files
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <style>
        body {
            background: linear-gradient(135deg, #667eea, #764ba2);
            font-family: Arial;
        }
        .container {
            width: 450px;
            margin: 80px auto;
            background: white;
            padding: 25px;
            border-radius: 12px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-bottom: 12px;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .result {
            margin-top: 15px;
            padding: 12px;
            text-align: center;
            font-weight: bold;
            border-radius: 8px;
        }
        .churn {
            background: #ffe5e5;
            color: red;
        }
        .no-churn {
            background: #e6fffa;
            color: green;
        }
    </style>
</head>

<body>
<div class="container">
    <h2> Customer Churn Prediction</h2>

    <form method="post">
        <label>Tenure (months)</label>
        <input type="number" name="tenure" required>

        <label>Monthly Charges</label>
        <input type="number" step="0.01" name="MonthlyCharges" required>

        <label>Total Charges</label>
        <input type="number" step="0.01" name="TotalCharges" required>

        <label>Contract Type</label>
        <select name="Contract">
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
        </select>

        <button type="submit">Predict Churn</button>
    </form>

    {% if prediction %}
        <div class="result {{ css }}">
            {{ prediction }}
        </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    css = None

    if request.method == "POST":
        # Get user input
        tenure = int(request.form["tenure"])
        monthly = float(request.form["MonthlyCharges"])
        total = float(request.form["TotalCharges"])
        contract = request.form["Contract"]

        # Create input dictionary
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0
        }

        # Convert to DataFrame with all columns
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if result == 1:
            prediction = f" Customer likely to CHURN (Risk: {prob*100:.1f}%)"
            css = "churn"
        else:
            prediction = f" Customer likely to STAY (Risk: {prob*100:.1f}%)"
            css = "no-churn"

    return render_template_string(
        HTML,
        prediction=prediction,
        css=css
    )

if __name__ == "__main__":
    app.run(debug=True)
