from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
from io import BytesIO

from IPython.display import display




client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Dy8Ow3DQB5W7WtvMMTML"
)


result = client.run_workflow(
    workspace_name="ocr-pwajs",
    workflow_id="custom-workflow",
    images={
        "image": "/content/20250403_115525.jpg" # sample image 
    },
    use_cache=True
)

print(result)


