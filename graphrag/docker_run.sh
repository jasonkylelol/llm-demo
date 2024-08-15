IMAGE=graphrag:v0.2.2-py310-v1.0

docker run -it --rm --gpus all -h graph-rag --name graph-rag \
    -p 38062:80 \
    -v /data/lm/github/llm-demo/graphrag/serving.py:/workspace/serving.py \
    -v /data/lm/github/llm-demo/graphrag/index.py:/workspace/index.py \
    -v /data/lm/github/llm-demo/graphrag/ragtest:/workspace/ragtest \
    $IMAGE bash
