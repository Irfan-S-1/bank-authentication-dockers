# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:32:28 2020

@author: Asus
"""

import os
os.getcwd()  
os.chdir('D:\Data Science\Project\Project Bank note')

from flask import Flask,request
import pandas as pd
import numpy as np


import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_in=open("classifier.pkl","rb")



@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    """ Lets Authenticate the Bank note's
    ---
    parameters:
        - name:variance
          in:query
          type:number
          required:true
        - name:skewness
          in:query
          type:number
          required:true
        - name:curtosis
          in:query
          type:number
          required:true
        - name:entropy
          in:query
          type:number
          required:true
          
          response:
              20:
                  description:The output values
                  
      """
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=pickle_in.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "The predicted values is" +str(prediction)


if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000)
    

    
