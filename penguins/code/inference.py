
import os
import json
import boto3
import requests
import numpy as np
import pandas as pd

from pickle import load
from pathlib import Path


PIPELINE_FILE = Path("/tmp") / "pipeline.pkl"
CLASSES_FILE = Path("/tmp") / "classes.csv"

s3 = boto3.resource("s3")


def handler(data, context):
    """
    This is the entrypoint that will be called by SageMaker when the endpoint
    receives a request. You can see more information at 
    https://github.com/aws/sagemaker-tensorflow-serving-container.
    """
    print("Handling endpoint request")
    
    instance = _process_input(data, context)
    output = _predict(instance, context)
    return _process_output(output, context)


def _process_input(data, context):
    print("Processing input data...")
    
    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we can use the
        # data directly.
        endpoint_input = data
    elif context.request_content_type in ("application/json", "application/octet-stream"):
        # When the endpoint is running, we will receive a context
        # object. We need to parse the input and turn it into 
        # JSON in that case.
        endpoint_input = json.loads(data.read().decode("utf-8"))

        if endpoint_input is None:
            raise ValueError("There was an error parsing the input request.")
    else:
        raise ValueError(f"Unsupported content type: {context.request_content_type or 'unknown'}")
        
    return _transform(endpoint_input)


def _predict(instance, context):
    print("Sending input data to model to make a prediction...")
    
    model_input = json.dumps({"instances": [instance]})
    
    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we want to return
        # a fake prediction back.
        result = {
            "predictions": [
                [0.2, 0.5, 0.3]
            ]
        }
    else:
        # When the endpoint is running, we will receive a context
        # object. In that case we need to send the instance to the
        # model to get a prediction back.
        response = requests.post(context.rest_uri, data=model_input)
        
        if response.status_code != 200:
            raise ValueError(response.content.decode('utf-8'))
            
        result = json.loads(response.content)
    
    print(f"Response: {result}")
    return result


def _process_output(output, context):
    print("Processing prediction received from the model...")
    
    response_content_type = "application/json" if context is None else context.accept_header
    
    prediction = np.argmax(output["predictions"][0])
    confidence = output["predictions"][0][prediction]
    
    print(f"Prediction: {prediction}. Confidence: {confidence}")
    
    result = json.dumps({
        "species": _get_class(prediction),
        "prediction": int(prediction),
        "confidence": confidence
    }), response_content_type
    
    return result


def _get_pipeline():
    """
    This function will download the Scikit-Learn pipeline from S3 if it doesn't
    already exist. The function will use the `S3_LOCATION` environment
    variable to determine the location of the pipeline.
    """
    
    if not PIPELINE_FILE.exists():
        s3_uri = os.environ.get("S3_LOCATION", None)
        
        s3_parts = s3_uri.split('/', 3)
        bucket = s3_parts[2]
        key = s3_parts[3]

        s3.Bucket(bucket).download_file(f"{key}/pipeline.pkl", str(PIPELINE_FILE))
        
    return load(open(PIPELINE_FILE, 'rb'))


def _get_class(prediction):
    """
    This function returns the class name of a given prediction. 
    
    The function downloads the file with the list of classes from S3 if it doesn't
    already exist. The function will use the `S3_LOCATION` environment
    variable to determine the location of the file.
    """
    
    if not CLASSES_FILE.exists():
        s3_uri = os.environ.get("S3_LOCATION", None)
        
        s3_parts = s3_uri.split('/', 3)
        bucket = s3_parts[2]
        key = s3_parts[3]

        s3.Bucket(bucket).download_file(f"{key}/classes.csv", str(CLASSES_FILE))
            
    with open(CLASSES_FILE) as f:
        file = f.readlines()
        
    classes = list(map(lambda x: x.replace("'", ""), file[0].split(',')))
    return classes[prediction]


def _transform(payload):
    """
    This function transforms the payload in the request using the
    Scikit-Learn pipeline that we created during the preprocessing step.
    """
    
    print("Transforming input data...")

    island = payload.get("island", "")
    culmen_length_mm = payload.get("culmen_length_mm", 0)
    culmen_depth_mm = payload.get("culmen_depth_mm", 0)
    flipper_length_mm = payload.get("flipper_length_mm", 0)
    body_mass_g = payload.get("body_mass_g", 0)
    
    data = pd.DataFrame(
        columns=["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], 
        data=[[
            island, 
            culmen_length_mm, 
            culmen_depth_mm, 
            flipper_length_mm, 
            body_mass_g
        ]]
    )
    
    result = _get_pipeline().transform(data)
    return result[0].tolist()
