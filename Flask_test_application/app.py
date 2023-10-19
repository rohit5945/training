#imports required for flask part
from flask import Flask, render_template,request

# Initialize the model
from model import TwitterClassification

app = Flask(__name__)
model_path = f"saved_model/twitter_sentment"
@app.route("/")
def msg():
    return render_template('index.html')

@app.route("/classify",methods=['POST','GET'])
def getSummary():
    body=request.form['data']
    class_model = TwitterClassification(model_path)
    result = class_model.predict(body)
    return render_template('classification.html',result=result)
    #return result

if __name__ =="__main__":
    app.run(debug=True,port=8000)