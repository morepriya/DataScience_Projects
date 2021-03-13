import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('Status'))
        try:
            Status=float(request.form['Status'])
            Schooling=float(request.form['Schooling'])
            Income_Comp_Of_Resources=float(request.form['Income_Comp_Of_Resources'])
            HIVorAIDS=float(request.form['HIV/AIDS'])
            Adult_Mortality=float(request.form['Adult_Mortality'])
            Alcohol=float(request.form['Alcohol'])
            pred_args= [Status,Schooling,Income_Comp_Of_Resources,HIVorAIDS,Adult_Mortality,Alcohol]
            pred_args_arr = np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            model_prediction =model.predict(pred_args_arr)
            model_prediction=round(float(model_prediction),2)
            
        except ValueError:
            return "Please check if the values are entered correctly" 
    return render_template('index.html',prediction_text='Predicted Life Expectancy = {}'.format(model_prediction))
            
     
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict([[final_features]])
    
    #output = round(prediction[0], 2)
    #return render_template('index.html', prediction_text='Predicted Life Expectancy ${}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)