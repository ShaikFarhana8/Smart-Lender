import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("rdf.pkl", "rb"))

# Load scaler if available
scaler = None
if os.path.exists("scale1.pkl"):
    scaler = pickle.load(open("scale1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/input")
def input_page():
    return render_template("input.html")


@app.route("/submit", methods=["POST"])
def submit():

    try:

        gender = int(request.form["Gender"])
        married = int(request.form["Married"])
        dependents = int(request.form["Dependents"])
        education = int(request.form["Education"])
        self_employed = int(request.form["Self_Employed"])
        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])
        credit_history = int(request.form["Credit_History"])
        property_area = int(request.form["Property_Area"])
        # Validate Inputs
        if applicant_income <= 0:
           return "Applicant Income must be greater than 0"
        if loan_amount <= 0:
            return "Loan Amount must be greater than 0"
        if loan_term <= 0:
            return "Loan Term must be greater than 0"

        features = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]])

        if scaler is not None:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1] * 100
            # ===============================
            # Loan Eligibility Score
            # ===============================
            score = round(probability)
            # ===============================
            # # Risk Level
            # # ===============================
            if probability >= 85:
               risk = "🟢 Very Low Risk"
            elif probability >= 70:
               risk = "🟢 Low Risk"
            elif probability >= 50:
                risk = "🟡 Medium Risk"
            else:
                risk = "🔴 High Risk"
        else:
            probability = 100 if prediction == 1 else 0

        if prediction == 1:
            result = "Loan Approved ✅"
            message = "Congratulations! Based on your financial profile, your loan has a high probability of approval."
        else:
            result = "Loan Rejected ❌"
            message = "Unfortunately, your loan is not likely to be approved at this time. Improving your credit history or financial profile may increase your chances."
        # Approval Score
        score = round(probability)

        # -------------------------
        # Risk Level
        # -------------------------

        if probability >= 90:
            risk = "🟢 Excellent"
        elif probability >= 75:
            risk = "🟢 Low Risk"
        elif probability >= 60:
            risk = "🟡 Medium Risk"
        elif probability >= 40:
            risk = "🟠 High Risk"
        else:
            risk = "🔴 Very High Risk"
            # Loan Eligibility Score
            score = round(probability)
     # EMI Calculation
        annual_interest_rate = 8.5
        monthly_interest_rate = annual_interest_rate / (12 * 100)

        loan_amount_rupees = loan_amount
        months = int(loan_term)

        if monthly_interest_rate > 0:
          emi = (
                loan_amount_rupees
                * monthly_interest_rate
                * (1 + monthly_interest_rate) ** months
            ) / (
              (1 + monthly_interest_rate) ** months - 1
            )
        else:
           emi = loan_amount_rupees / months

        # -------------------------
        # Prediction Explanation
        # -------------------------

        reasons = []

        if credit_history == 1:
            reasons.append("✔ Good Credit History")
        else:
            reasons.append("❌ No Credit History")

        if applicant_income >= 5000:
            reasons.append("✔ Stable Applicant Income")
        else:
            reasons.append("⚠ Low Applicant Income")

        if coapplicant_income > 0:
            reasons.append("✔ Additional Co-applicant Income")

        if education == 1:
            reasons.append("✔ Graduate Applicant")
        else:
            reasons.append("⚠ Not Graduate")

        if self_employed == 0:
            reasons.append("✔ Salaried Employee")
        else:
            reasons.append("ℹ Self-Employed Applicant")
            if dependents == 0:
                reasons.append("✔ No Dependents")
            elif dependents <= 2:
                reasons.append("✔ Few Dependents")
            else:
                reasons.append("⚠ Many Dependents")

        if loan_amount <= 150:
            reasons.append("✔ Moderate Loan Amount")
        else:
            reasons.append("⚠ High Loan Amount")

        if property_area == 2:
            reasons.append("✔ Urban Property Area")
        elif property_area == 1:
            reasons.append("✔ Semiurban Property Area")
        else:
            reasons.append("✔ Rural Property Area")
        current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
        record = {
    "Prediction": result,
    "Probability": probability,
    "Risk": risk,
    "Applicant Income": applicant_income,
    "Loan Amount": loan_amount
}
        data = pd.DataFrame([record])
        data.to_csv(
    "history.csv",
    mode="a",
    header=not os.path.exists("history.csv"),
    index=False
)

        return render_template(
    "output.html",
    score=score,
    prediction=result,
    probability=round(probability, 2),
    risk=risk,
    emi=round(emi, 2),
    reasons=reasons
    time=current_time
)
            
        
    except Exception as e:
        return f"""
        <h2>Error Occurred</h2>
        <p>{e}</p>
        <a href='/input'>Go Back</a>
        """


if __name__ == "__main__":
    app.run(debug=True)