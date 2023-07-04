import json

def preprocess_handler(inference_record):
    input_data = inference_record.endpoint_input.data
    output_data = json.loads(inference_record.endpoint_output.data)
    
    response = json.loads(input_data)
    response["groundtruth"] = output_data["prediction"]
    return response
