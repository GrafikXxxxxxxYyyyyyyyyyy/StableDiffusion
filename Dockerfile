FROM runpod/base:0.4.2-cuda11.8.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Maybe create and activate .venv
RUN python3 -m venv .venv && \ 
    source .venv/bin/activate 

# Install Python dependencies
COPY ./requirements.txt /app/requirements.txt

RUN python3.11 -m pip install pip && \
    python3.11 -m pip install -r /app/requirements.txt --no-cache-dir && \
    rm /app/requirements.txt

RUN pip install hf_transfer

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Cache checkpoints
RUN python3.11 -c "import torch; from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('OnMoon/sdxl_Juggernaut', torch_dtype=torch.float16, variant='fp16')"
RUN python3.11 -c "import torch; from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('OnMoon/sdxl_AnimaPencil', torch_dtype=torch.float16, variant='fp16')"
RUN python3.11 -c "import torch; from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('OnMoon/sdxl_AutismMix', torch_dtype=torch.float16, variant='fp16')"

# Cache LoRAs
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_tsunade.safetensors')"
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_melanie.safetensors')"
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_makima.safetensors')"
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_frieren.safetensors')"
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_ciri.safetensors')"
RUN python3.11 -c "import torch; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='OnMoon/loras', filename='sdxl_briana.safetensors')"


# Cache Models
COPY ./models /app/models
COPY ./pipelines /app/pipelines
COPY ./runpod-worker /app/runpod-worker

WORKDIR /app

# Set permissions and specify the command to run
RUN chmod +x /app/runpod-worker/start.sh
CMD /app/runpod-worker/start.sh