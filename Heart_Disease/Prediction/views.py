from django.shortcuts import render
import joblib
import os

# Get the absolute path of the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
model_path = os.path.join(BASE_DIR, 'Model', 'heart_model.pkl')
model = joblib.load(model_path)


def predict(request):
    result = None
    probability = None

    if request.method == "POST":
        try:
            # Get all 13 input features
            age = float(request.POST.get('age', 0))
            sex = float(request.POST.get('sex', 0))
            cp = float(request.POST.get('cp', 0))
            trestbps = float(request.POST.get('trestbps', 0))
            chol = float(request.POST.get('chol', 0))
            fbs = float(request.POST.get('fbs', 0))
            restecg = float(request.POST.get('restecg', 0))
            thalach = float(request.POST.get('thalach', 0))
            exang = float(request.POST.get('exang', 0))
            oldpeak = float(request.POST.get('oldpeak', 0))
            slope = float(request.POST.get('slope', 0))
            ca = float(request.POST.get('ca', 0))
            thal = float(request.POST.get('thal', 0))

            # Prepare all 13 features
            X = [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]]

            # Make prediction
            prediction = model.predict(X)[0]

            # Prediction result
            if prediction == 1:
                result = "Heart Disease"
            else:
                result = "No Heart Disease"

            # Get probability if available
            if hasattr(model, "predict_proba"):
                probability = round(model.predict_proba(X)[0][1] * 100, 2)

        except Exception as e:
            result = f"Error: {e}"

    return render(
        request,
        "index.html",
        {
            "result": result,
            "probability": probability
        }
    )