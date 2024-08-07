from flask import Flask, render_template,request     #to create web application
import numpy as np                              #used for working with arraysand also for numercal values as it is easy to calculation
import joblib                      #to perform operation in parallel on datasets
app=Flask(__name__)                 #creates a Flask application object --app-- in current python module

@app.route('/predict',methods=['GET','POST'])    # does the main job of allowing us to predictions
def predict():
    if request.method=='POST':
        n=float(request.form['n'])
        p=float(request.form['p'])
        k=float(request.form['k'])
        temp=float(request.form['temp'])
        humidity=float(request.form['humidity'])
        ph=float(request.form['ph'])
        
        rainfall=float(request.form['rainfall'])
        
        testdata=np.array([[n,p,k,temp,humidity,ph,rainfall]])
        model=joblib.load("knncropmodel.pkl")
        res=model.predict(testdata)
        cropname=res[0]
        print(f"predicted result={cropname}")
        
        return render_template('index.html',result=cropname)  #generate output from jinja2 template file
@app.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':       #indicates this program is the main program to be executed
        app.run(debug=True)      #enables debugmode