# importing the packages
import json
# import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, flash#, redirect, url_for, session, logging
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# value of __name__ should be  '__main__'
app = Flask(__name__,static_url_path="")
# Loading model so that it works on production
model = joblib.load('./model/model.pkl')


@app.route('/')
def index():
	# Index page
	return render_template('index.html')

@app.route('/problem')
def problem():
	# about me page
	return render_template('problem.html')
@app.route('/satisfaction')	
def satisfaction():
	#commitment
	return render_template('satisfaction.html')	

class PredictorsForm(Form):
	"""
	This is a form class to retrieve the input from user through form

	Inherits: request.form class
	"""
	satisfaction_level = StringField(u'Satisfaction Level (Valid Values: 0.00-1.00)', validators=[validators.input_required()])
	last_evaluation = StringField(u'Commitment (Valid Values: 0.00-1.00)', validators=[validators.input_required()])
	number_project = StringField(u'Number of Projects(Eg.:0,1,2,...)', validators=[validators.input_required()])
	avg_monthly_hours = StringField(u'Average Monthly Hours (Eg.: 0,1,2,...)', validators=[validators.input_required()])
	time_spent = StringField(u'Time Spent in Organisation (In Years)', validators=[validators.input_required()])
	work_accident = StringField(u'Work Accident (Yes:1,No:0)', validators=[validators.input_required()])
	promotion = StringField(u'Promotion In Last 5 years (Yes:1,No:0)', validators=[validators.input_required()])
	salary = StringField(u'Salary (High:0,Low:1,Medium:2)', validators=[validators.input_required()])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	form = PredictorsForm(request.form)
	
	# Checking if user submitted the form and the values are valid
	if request.method == 'POST' and form.validate():
		# Now save all values passed by user into variables
		satisfaction_level = form.satisfaction_level.data
		last_evaluation = form.last_evaluation.data
		number_project = form.number_project.data
		avg_monthly_hours = form.avg_monthly_hours.data
		time_spent = form.time_spent.data
		work_accident = form.work_accident.data
		promotion = form.promotion.data
		salary = form.salary.data

		# Creating input for model for predictions
		predict_request = [float(satisfaction_level), float(last_evaluation), int(number_project), int(avg_monthly_hours), int(time_spent), int(work_accident), int(promotion), int(salary)]
		predict_request = np.array(predict_request).reshape(1, -1)

		# Class predictions from the model
		prediction = model.predict(predict_request)
		prediction = str(prediction[0])

		# Survival Probability from the model
		predict_prob = model.predict_proba(predict_request)
		predict_prob = str(predict_prob[0][1])

		# Passing the predictions to new view(template)
		return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob,inputs={
                        "satisfaction_level":satisfaction_level,
                       "last_evaluation":last_evaluation,
                       "number_project":number_project,
                       "avg_monthly_hours":avg_monthly_hours,
                    "time_spent":time_spent,
                    "work_accident":work_accident,
                    "promotion":promotion,
                    "salary":salary    })

	return render_template('predict.html', form=form)

@app.route('/train', methods=['GET'])
def train():
	# reading data
	df = pd.read_csv("./data/HR.csv")

	#defining predictors and label columns to be used
	predictors = ['satisfaction_level','last_evaluation','number_project','avg_monthly_hours','time_spent','work_accident','promotion','salary']
	label = 'left'

	#Encoding Categorical Data(Salary)
	from sklearn.preprocessing import LabelEncoder
	labelencoder_X = LabelEncoder()
	predictors[:, 7] = labelencoder_X.fit_transform(predictors[:, 7])
	
	#Splitting data into training and testing
	df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)


	# Initializing the model
	model = RandomForestClassifier(n_estimators=25, random_state=42)

	# Fitting the model with training data
	model.fit(X=df_train, y=y_train)

	# Saving the trained model on disk
	joblib.dump(model, './model/model.pkl')

	# Return success message for user display on browser
	return 'Success'

if __name__ == '__main__':
	# Load the pre-trained model from the disk
	# model = joblib.load('./model/model.pkl')
	# Running the app in debug mode allows to change the code and
	# see the changes without the need to restart the server
	app.run(debug=True)
