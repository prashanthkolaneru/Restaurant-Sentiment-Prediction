from flask import Flask,jsonify,request,render_template
import pickle
from processSentence import process_features
import pandas as pd

app=Flask(__name__)

#load pickle file 
model = pickle.load(open('Resturant.pkl','rb'))

Data = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t')

@app.route("/")

def index():
        return render_template('index.html')

@app.route("/",methods=["POST"])
def predict():

         
        msg=request.form['msg']       
        Data['Review'][0]=msg
        
        input=process_features(Data['Review'])

        
        result=model.predict(input[0])
        if result[0]==1:
             myresult="Recomended"
        else:
             myresult="Not Recomended"


        res=myresult



        #return jsonify(res)
        return render_template('index.html', prediction_text=' The Restaurant is  {}'.format(res))

    

if __name__=='__main__':
      
       app.run(debug=True)
