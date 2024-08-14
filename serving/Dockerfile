from pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

workdir /workspace

add models models
add requirements.txt requirements.txt
run pip install -r requirements.txt

add openai_api_all_in_one.py openai_api_all_in_one.py
add openai_api_server_glm4.py openai_api_server_glm4.py
add openai_api_embedding_app.py openai_api_embedding_app.py

env MODEL_ROOT=/workspace/models
env LLM_MODEL_PATH=/workspace/models/llm
env EMBEDDING_MODEL_PATH=/workspace/models/embedding

CMD ["python", "openai_api_all_in_one.py"]

# docker build -t graphrag/openai-compatible-api:v1.0 .
# docker run -it --rm --gpus device=3 -e SERVER_PORT=8062 -p 38062:8062 graphrag/openai-compatible-api:v1.0