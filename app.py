from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlFlowProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("dvc repro")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user from the form
            surname = request.form['Surname']
            credit_score = int(request.form['creditScore'])
            geography = request.form['geography']
            gender = request.form['gender']
            age = int(request.form['age'])
            tenure = int(request.form['tenure'])
            balance = float(request.form['balance'])
            number_of_products = int(request.form['numberOfProducts'])
            credit_card = int(request.form['creditCard'])
            active_member = int(request.form['activeMember'])
            estimated_salary = float(request.form['estimatedSalary'])

            # Include rowNumber and customerId for data schema sake
            rowNumber, customerId = 0, 0
            
            field_names = [
                'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]

            # Creating a data array
            data = [
                rowNumber, customerId, surname, credit_score, geography, gender, age, tenure, 
                balance, number_of_products, credit_card, active_member, estimated_salary
            ]
            matrix = np.array(data).reshape(1, -1)

            data = pd.DataFrame(matrix, columns=field_names)

            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)

            if prediction[0] == 0:
                prediction = 'No'
            else:
                prediction = 'Yes'

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return str(e)
    return render_template('index.html')



if __name__ == "__main__":
	app.run(debug=True)