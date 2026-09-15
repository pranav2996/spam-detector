from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route('/')
def home():
    return render_template(
        'index.html',
        prediction_text=None,
        message=""
    )


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    # Updated result labels
    result = (
        "Phishing Message ❌"
        if prediction == 1
        else "Not a Phishing Message ✅"
    )

    return render_template(
        'index.html',
        prediction_text=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
