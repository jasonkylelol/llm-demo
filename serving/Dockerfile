from pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

workdir /workspace

add models models
add requirements.txt requirements.txt
run pip install -r requirements.txt

add openai_api_all_in_one.py openai_api_all_in_one.py
add openai_api_glm4_app.py openai_api_glm4_app.py
add openai_api_qwen2_app.py openai_api_qwen2_app.py
add openai_api_embedding_app.py openai_api_embedding_app.py

env MODEL_ROOT=/workspace/models
env LLM_MODEL_PATH=/workspace/models/llm
env EMBEDDING_MODEL_PATH=/workspace/models/embedding

CMD ["python", "openai_api_all_in_one.py"]

# docker build -t graphrag/openai-compatible-api:v2.0 .
# docker run -it --rm --gpus device=3 -e SERVER_PORT=8063 -p 38063:8063 graphrag/openai-compatible-api:v2.0
