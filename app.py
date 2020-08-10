from chalice import Chalice, BadRequestError
import base64, os, boto3, ast
import numpy as np
import json
from urllib.request import urlopen

# chalice app configuration
app = Chalice(app_name='image-classifier')
app.debug=True

# Get environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
sm_runtime = boto3.client('sagemaker-runtime')

# RESTful endpoint
@app.route('/', methods=['POST'])
def index():
    body = app.current_request.json_body
    image = base64.b64decode(body['data'])

    # Invoking SageMaker Endpoint
    response = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/x-image',
        Body=image)

    propa_result = json.loads(response['Body'].read().decode())
    
    # Load names for image classes
    text_url  = urlopen("https://raw.githubusercontent.com/a-shlash/sagemaker_chalice_app/master/"
                        "imagenet1000_clsidx_to_labels.txt")

    image_categories = {}
    for line in text_url:
    	decoded_line = line.decode("utf-8")
    	key, val = decoded_line.strip().split(':')
    	image_categories[key] = val
    	
    result = "Result: \n label: " + image_categories[str(np.argmax(propa_result))]+ " \n probability: " + str(np.amax(propa_result))
    return(result)