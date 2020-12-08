from flask import Flask, jsonify, request
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# set the do_debug to run debug mode in local
do_debug = False

# create Flask object
api = Flask(__name__)

# create  api.route
# each route is associated with a function

# สร้าง Home Page
@api.route('/') 
def main():
    return 'Hello This is home page'


# what we can return
# - string
# - dictionary
# - list (import jsonify)
# - webpage (import render_template)


# Receive input from users
@api.route('/pred')
def get_param():
    userFollowerCount = request.args.get('userFollowerCount', default=0, type=int)
    userFollowingCount = request.args.get('userFollowingCount', default=0, type=int)
    userBiographyLength = request.args.get('userBiographyLength', default=0, type=int)
    userMediaCount = request.args.get('userMediaCount', default=0, type=int)
    userHasProfilPic = request.args.get('userHasProfilPic', default=0, type=int)
    userIsPrivate = request.args.get('userIsPrivate', default=0, type=int)
    usernameDigitCount = request.args.get('usernameDigitCount', default=0, type=int)

    data = [[userFollowerCount, userFollowingCount, userBiographyLength, userMediaCount, userHasProfilPic, userIsPrivate, usernameDigitCount]]
    model_file = 'fake_predict.sav'
    output = get_prediction(data, model_file)
    return jsonify({'result': output})


# function to load model and make prediction
def get_prediction(data, model_file):
  # load the model from disk
  loaded_model = joblib.load(model_file)

  col_names = ['userFollowerCount', 'userFollowingCount', 'userBiographyLength',
              'userMediaCount', 'userHasProfilPic', 'userIsPrivate',
              'usernameDigitCount']
  df_data = pd.DataFrame(data, columns=col_names)

  pred = loaded_model.predict(df_data)[0]

  output = 'fake account' if pred =='1' else 'real account'
  return output


if __name__ == '__main__':
    # the following will run only if we run this script directly 
    # (i.e. by running 'python api_instafake.py') 

    #api.run()

    # if you want the auto update
    #api.run(debug=True)
    api.run(debug=do_debug)


# # to run this file
# python api_instafake.py

# now you can try this on your browser 
# http://127.0.0.1:5000/pred?userFollowerCount=204&userFollowingCount=445&userBiographyLength=118&userMediaCount=92&userHasProfilPic=1&userIsPrivate=1&usernameDigitCount=0
# notice the route is followed by ?
# use & to specify more parameters


