import pandas as pd
import pickle
import numpy as np
from flask import Flask,request,app,render_template,jsonify,url_for

app=Flask(__name__)
#Load the model
reg_pred = pickle.load(open('iris.pkl','rb'))
 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred_api',methods=['POST'])
def pred_api():
    data =request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = reg_pred.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/pred',methods=['POST'])
def pred():
    data = [float(i) for i in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = reg_pred.predict(final_input)[0]
    if output ==0:
        o = "Iris-Setosa"
    elif output == 1:
        o="Iris-Versicolour" 
    else:
        o="Iris-Virginica"   
    return render_template('home.html',pred_text="The predicted Flower is of type {}".format(o))


if __name__=="__main__":
    app.run(debug=True)