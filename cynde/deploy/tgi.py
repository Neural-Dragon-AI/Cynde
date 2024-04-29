import os
import subprocess
from pathlib import Path
from pydantic import BaseModel, conint, ValidationError
from typing import List, Optional
from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
   
]

class Prompt(BaseModel):
    system_prompt: str
    user_message: str
    output_schema: Optional[dict] = None
    repetition_penalty: Optional[float] = None

def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
           
        ],
    )


app = App(
    "example-tgi-" + MODEL_ID.split("/")[-1]
)  # Note: prior to April 2024, "app" was called "stub"
print("App name:", app.name)
latest_version = "ghcr.io/huggingface/text-generation-inference:sha-f9cf345"
version_from_example = "ghcr.io/huggingface/text-generation-inference:1.4"
tgi_image = (
    Image.from_registry(latest_version)
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .run_commands("pip install pydantic --upgrade")
    .pip_install("outlines")
    .pip_install("text-generation")
)

GPU_CONFIG = gpu.A100(count=1,size="40GB")  # 2 H100s


@app.cls(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        
        
        self.template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

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
    async def generate(self, prompt: Prompt) :
        schema = prompt.output_schema
        repetition_penalty = prompt.repetition_penalty
        prompt = self.template.format(system_prompt=prompt.system_prompt,user_message=prompt.user_message)

        
        if schema is None:
            result = await self.client.generate(
                prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
            )
        else:
            
            result = await self.client.generate(
                prompt, max_new_tokens=1024, decoder_input_details=False,
        seed=1, stop_sequences=["<|eot_id|>"], repetition_penalty=repetition_penalty,grammar={
            "type": "json",
            "value": schema
        }
            )

        return result

    @method()
    async def generate_stream(self, question: str):
        prompt = self.template.format(user=question)

        async for response in self.client.generate_stream(
            prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
        ):
            if (
                not response.token.special
                and response.token.text != "<|eot_id|>"
            ):
                yield response.token.text



    

