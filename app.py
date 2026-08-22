from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, message="")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    result = "Spam ❌" if prediction == 1 else "Not Spam ✅"

    return render_template(
        'index.html',
        prediction_text=result,
        message=message   # 👈 keep message
    )

if __name__ == "__main__":
    app.run(debug=True)