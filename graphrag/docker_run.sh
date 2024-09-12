IMAGE=graphrag:v0.3.2-py310-v2.0

docker run -it --rm --gpus all -h graph-rag --name graph-rag \
    -p 38062:80 \
    -v /data/lm/github/llm-demo/graphrag/serving.py:/workspace/serving.py \
    -v /data/lm/github/llm-demo/graphrag/index.py:/workspace/index.py \
    -v /data/lm/github/llm-demo/graphrag/ragtest:/workspace/ragtest \
    -v /data/lm/github/llm-demo/graphrag/graphrag:/workspace/graphrag \
    $IMAGE bash
