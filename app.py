# importing required libraries 

from flask import Flask, request,render_template
import pickle
import pandas as pd




 

app=Flask(__name__)





# importing the pickle file that we generated in the backend (main source code) 

pickle_in = open("selected_model.pkl","rb")
classifier=pickle.load(pickle_in)




# creating the routes

@app.route('/')
def home():
    return render_template('model.html')



# this is the first route where user can input features values manually 
# and will get the predicted output (= 0,1)
# where 0 means forged and 1 means authentic

@app.route('/Input Individual Values',methods=["post"])
def predict_note_authentication():
    '''
    For rendering results on HTML GUI
    '''
    
    features = [x for x in request.form.values()]
    
    variance = features[0]
    
    skewness = features[1]
    
    kurtosis = features[2]
    
    entropy = features[3]
    
    prediction = classifier.predict([[variance,skewness,kurtosis,entropy]])
    
    
    
    return render_template('model.html', prediction_text1 = 'The prediction (0 means forged and 1 means authentic) of the input data is : {}'.format(prediction))


 # this is the second route where user can input features values by uploading 
# a single csv file and will get the predicted outputs (= 0,1) in the list format 
# where 0 means forged and 1 means authentic 


@app.route('/input_dataset',methods=["POST"])
def predict_bank_note_file():
    
    '''
    For rendering results on HTML GUI
    '''
    #store the file contents as a string
    f = request.files['myfile']

    
    
    test_df = pd.read_csv(f)
    
    
    prediction = classifier.predict(test_df)
    
    return render_template('model.html', prediction_text2 = 'The prediction (0 means forged and 1 means authentic) of the input dataset(.csv file) is :  {}'.format(list(prediction)))


 
 
 
 
 




if __name__=='__main__':
    app.run(debug = True)

