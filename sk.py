from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
#from sagemaker_sklearn_extension.externals import read_csv_data

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'State',
    'Account Length',
    'Area Code',
    'Phone',
    "Int'l Plan",
    'VMail Plan',
    'VMail Message',
    'Day Mins',
    'Day Calls',
    'Day Charge',
    'Eve Mins',
    'Eve Calls',
    'Eve Charge',
    'Night Mins',
    'Night Calls',
    'Night Charge',
    'Intl Mins',
    'Intl Calls',
    'Intl Charge',
    'CustServ Calls'] # after being dried

label_column = 'Churn?'

feature_columns_dtype = {
    'State' :  str,
    'Account Length' :  np.int64,
    'Area Code' :  str,
    'Phone' :  str,
    "Int'l Plan" :  str,
    'VMail Plan' :  str,
    'VMail Message' :  np.int64,
    'Day Mins' :  np.float64,
    'Day Calls' :  np.int64,
    'Day Charge' :  np.float64,
    'Eve Mins' :  np.float64,
    'Eve Calls' :  np.int64,
    'Eve Charge' :  np.float64,
    'Night Mins' :  np.float64,
    'Night Calls' :  np.int64,
    'Night Charge' :  np.float64,
    'Intl Mins' :  np.float64,
    'Intl Calls' :  np.int64,
    'Intl Charge' :  np.float64,
    'CustServ Calls' :  np.int64}

label_column_dtype = {'Churn?': str} # +1.5 gives the age in years

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def _is_inverse_label_transform():
    """Returns True if if it's running in inverse label transform."""
    return os.getenv('TRANSFORM_MODE') == 'inverse-label-transform'

def _is_feature_transform():
    """Returns True if it's running in feature transform mode."""
    print("===+++")
    print(os.environ['SAGEMAKER_REGION'])
    print(os.environ['TRANSFORM_MODE'])
    print(os.getenv('TRANSFORM_MODE'))
    return os.getenv('TRANSFORM_MODE') == 'feature-transform'


if __name__ == '__main__':
    print("===main===")
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--transform_mode', type=str, default=os.environ['SM_HP_TRANSFORM_MODE'])


    args = parser.parse_args()
    
    #if _is_feature_transform():
        # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)) for file in input_files ]
    concat_data = pd.concat(raw_data)

    numeric_features = list([
    'Account Length',
    'VMail Message',
    'Day Mins',
    'Day Calls',
    'Eve Mins',
    'Eve Calls',
    'Night Mins',
    'Night Calls',
    'Intl Mins',
    'Intl Calls',
    'CustServ Calls'])

    #numeric_features.remove(['State', 'Area Code','Phone' ,'Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['State','Area Code',"Int'l Plan",'VMail Plan']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder="drop")

    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, request_content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    
    
    print("=== input: feature transform====")
    content_type = request_content_type.lower(
    ) if request_content_type else "text/csv"
    content_type = content_type.split(";")[0].strip()
    
    print(content_type)
    print(input_data)
    
    print("****** It is input type")
    print(type(input_data))
#     print("****** It is input_data.encode() type")
#     print(type(input_data.encode()))
#     test_buf = input_data.encode()
#     print("****** It is str(byte_buffer,'utf-8')) type")
#     print(type(str(test_buf,'utf-8')))
    
    if isinstance(input_data, str):
        print("It is string")
        #byte_buffer = input_data.encode()
        str_buffer = input_data
    else:
        print("It is byte")
        str_buffer = str(input_data,'utf-8')
    
    print(str_buffer)
#     s=str(byte_buffer,'utf-8')
#     print(s)
    
    if _is_feature_transform():
        if content_type == 'text/csv':
            # Read the raw input data as CSV.
            df = pd.read_csv(StringIO(input_data),  header=None)
            print(df.head())
            if len(df.columns) == len(feature_columns_names) + 1:
                # This is a labelled example, includes the ring label
                df.columns = feature_columns_names + [label_column]
            elif len(df.columns) == len(feature_columns_names):
                # This is an unlabelled example.
                df.columns = feature_columns_names
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))
    
    
    if _is_inverse_label_transform():
        if (content_type == 'text/csv' or content_type == 'text/csv; charset=utf-8'):
            # Read the raw input data as CSV.
#             val = read_csv_data(source=byte_buffer)

#             print(val.head())
#             return val
            df = pd.read_csv(StringIO(str_buffer),  header=None)
            print(df.head())
            logging.info(f"Shape of the requested data: '{df.shape}'")
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))
    print("=== end input====")
def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    
    print("=== Output: feature transform====")
    print(type(prediction))
    if type(prediction) is not np.ndarray:
        prediction=prediction.toarray()
    
    print(accept)
    accept = 'text/csv'
    print(prediction)
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """

    
    if _is_feature_transform():
        print("=== prediction: feature transform====")
        features = model.transform(input_data)


        if label_column in input_data:
            # Return the label (as the first column) and the set of features.
            return np.insert(features.toarray(), 0, pd.get_dummies(input_data[label_column])['True.'], axis=1)
        else:
            # Return only the set of features
            return features

    
    
    
    if _is_inverse_label_transform():
        features = input_data.values
        return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    if _is_feature_transform():
        print("=== loading the model: feature transform====")
        preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
        return preprocessor