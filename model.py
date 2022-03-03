"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from preprocessing_methods import (
    split_time, 
    replace_valencia_pressure,
    handle_categorical_column_v2, 
    handle_colinear_temp_cols, 
    drop_columns
)
from print_helper import myprint

scaler = StandardScaler()

test_df = pd.read_csv('utils/data/df_test.csv')
split_test_df = split_time(test_df)

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    req = pd.DataFrame.from_dict([feature_vector_dict])

    # feature_vector_df = test_df
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # we load the test data as well a the request data, because some steps such as 
    # replacing missing values with averages, scaling and standardization can not be done if the dataframe has
    # only one row.

    # We split the time 
    req = split_time(req)
    req = replace_valencia_pressure(req, split_test_df)
    split_test_df = replace_valencia_pressure(split_test_df.copy(), split_test_df)
    req = handle_categorical_column_v2(req)
    split_test_df = handle_categorical_column_v2(split_test_df)
    req = req.drop(['time', 'Unnamed: 0'], axis=1)
    split_test_df = split_test_df.drop(['time', 'Unnamed: 0'], axis=1)
    columns = split_test_df.columns
    temp = scaler.fit_transform(pd.concat([split_test_df, req], ignore_index=True))
    temp = pd.DataFrame(temp).iloc[-1,:]
    temp = pd.DataFrame(temp).T
    temp.columns = columns
    req = handle_colinear_temp_cols(temp)
    req = drop_columns(req)
    # ------------------------------------------------------------------------

    return req # this should be of type dataframe

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    try: 
    # Data preprocessing.
        prep_data = _preprocess_data(data)
        # Perform prediction with model and preprocessed data.
        prediction = model.predict(prep_data)
        # Format as list for output standardisation.
        return prediction.tolist()
    except:
        return ["An error occured Please try again with different parameters"]
    
