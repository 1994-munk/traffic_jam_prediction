 # traffic_jam_prediction/app.py

from flask import Flask, request, render_template_string
import joblib
import numpy as np

# Load the trained model
model = joblib.load('traffic_prediction_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Homepage - form input
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        hour = int(request.form['hour'])
        junction = int(request.form['junction'])
        dayofweek = int(request.form['dayofweek'])
        rushhour = int(request.form['rushhour'])

        # Prepare data for prediction
        input_data = np.array([[hour, junction, dayofweek, rushhour]])
        prediction = model.predict(input_data)[0]

        # Show prediction result
        return render_template_string('''
            <h1>🚦 Traffic Prediction Result 🚦</h1>
            <h2>Expected Vehicles: {{ prediction }}</h2>
            <a href="/">Go back</a>
        ''', prediction=int(prediction))
    
    return render_template_string('''
        <h1>Hello Traffic Queen 👑🚦</h1>
        <form method="post">
            <label>Hour (0-23):</label><br>
            <input type="number" name="hour" required><br><br>

            <label>Junction (1-4):</label><br>
            <input type="number" name="junction" required><br><br>

            <label>Day of Week (0=Monday, 6=Sunday):</label><br>
            <input type="number" name="dayofweek" required><br><br>

            <label>Rush Hour (1=Yes, 0=No):</label><br>
            <input type="number" name="rushhour" required><br><br>

            <button type="submit">Predict Traffic 🚗</button>
        </form>
    ''')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
