# import imp
# from pyexpat import model
from flask import Flask, render_template,request
import pickle
import numpy as np
app= Flask(__name__)

# m= pickle.load(open("model_pickle","rb"))
with open("titanicaccuratemodel.pkl" , "rb") as f:
    mp = pickle.load(f)

@app.route("/")
def hello():
    return render_template("titanic.html")

@app.route("/predicts",methods=["POST","GET"])

def predicts():
    int_features= [int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction= mp.predict(final)

    if prediction == np.array([[1]]):
        return render_template("titanic.html", pred="You would not have died!")
    
    if prediction == np.array([[0]]):
        return render_template("titanic.html", pred="ALAS! You would have died!")


if __name__=="__main__":
    app.run(debug=True, threaded=True)

