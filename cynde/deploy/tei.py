import os
import subprocess
from pathlib import Path
from pydantic import BaseModel
from typing import List
from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method
from cynde.functional.embed.types import EmbeddingRequest

MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 512

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
]



def download_model():
    subprocess.run(
        [
            "text-embeddings-router",
            "download-weights",
            MODEL_ID,
        ],
    )

app = App("example-tei-" + MODEL_ID.split("/")[-1])
print("App name:", app.name)
latest_version = "ghcr.io/huggingface/text-embeddings-inference:1.2"
tei_image = (
    Image.from_registry(latest_version, add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []")
    
    .run_function(
        download_model,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .pip_install("httpx")
    .pip_install("numpy")
    .run_commands("pip install pydantic --upgrade")
    .apt_install("git")
    .pip_install("polars","scikit-learn","openai","tiktoken")#, force_build=True)
    
    .run_commands("git clone https://github.com/Neural-Dragon-AI/Cynde/")
    .env({"CYNDE_DIR": "/opt/cynde"})
    .run_commands("cd Cynde && pip install -r requirements.txt && pip install .")
    
    
)

GPU_CONFIG = gpu.A100(count=1,size="40GB")  
@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tei_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time
        from httpx import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-embeddings-router"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=60)

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Webserver ready!")

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def embed(self, request: EmbeddingRequest):
        import numpy as np
        import httpx

        try:
            response = await self.client.post("/embed", json=request.model_dump())
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            embeddings = np.array(response.json())
            return embeddings
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            # Handle the error appropriately, e.g., return an error response
        except Exception as e:
            print(f"An error occurred: {e}")
            # Handle the error appropriately, e.g., return an error response


@app.local_entrypoint()
def main():
    texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
    for text in texts:
        request = EmbeddingRequest(inputs=text)
        response = Model().embed.remote(request)
        print(response)
