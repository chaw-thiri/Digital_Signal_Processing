from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    api_key="Dy8Ow3DQB5W7WtvMMTML"
)

result = client.run_workflow(
    workspace_name="ocr-pwajs",
    workflow_id="custom-workflow",
    images={
        "image": "YOUR_IMAGE.jpg"
    }
)
