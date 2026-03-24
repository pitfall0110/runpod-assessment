## Large size model deploy on runpod serverless endpoint

1. Runpod allocates 5 workers to each serverless endpoint by default. (It can be 10, 15 or etc depending on the user credit)
2. At the cold start it is running Dockerfile and setup environment and run handler.py. At that time, as the model name  is setup in endpoint so that it is downloaded at the first time and can be for later use in caching.
3. In the handler, the endpoint is forced to use offline cache model for image generation. Then the handler loads pipeline and parse the input and generate output.
4. When the user sends request to serverless endpoint, then the idle worker becomes active and do the work.
Here are the expected input and output format.

    Expected input:
    ```json
    {
        "input": {
        "prompt": "a batman in a coffeeshop",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": 123
        }
    }
    ```

    Expected output:
    ```json
    {
        "status": "success",
        "prompt": prompt,
        "image_base64": image_b64,
        "width": width,
        "height": height,
        "seed": seed,
    }
    ```

5. The Runpod will be scaled down after an amount of time of inactivity and scale up again when user request comes in.

    Endpoint: https://api.runpod.ai/v2/unb8t0939pgc7i/run
