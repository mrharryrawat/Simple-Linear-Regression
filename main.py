from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("salary_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('yoexp_salary_dataset.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])

    # Cap experience between 0 and 10
    experience = max(0, min(experience, 10))

    # Predict salary
    prediction = float(model.predict([[experience]])[0])

    return render_template(
        'yoexp_salary_dataset.html',
        prediction=round(prediction, 2),
        experience=experience
    )

if __name__ == '__main__':
    app.run(debug=True)