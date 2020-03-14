from flask import Flask, render_template, url_for, jsonify, request
from Job_func_model import *


app = Flask(__name__)


dictt={
	'name':'mostafa',
	'age' : 24,
	'gender': 'male'
}
@app.route('/')
def home():
	return render_template('mytemp.html')

@app.route('/out',methods =['POST'])
def out():
	job_title = request.form['job']
	output = str(infer(job_title))
	return  render_template('mytemp.html', jobTitle=job_title+': ', result=output)




if __name__ == '__main__':
    app.run(debug=True)