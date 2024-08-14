IMAGE=graphrag:dev-0.1

docker run -it --rm --gpus all -h graph-rag --name graph-rag \
    -p 38062:80 \
    -e GRAPHRAG_API_BASE=http://192.168.0.20:38063/v1 \
    -v /data/lm/github/llm-demo/graphrag/serving.py:/workspace/serving.py \
    -v /data/lm/github/llm-demo/graphrag/index.py:/workspace/index.py \
    -v /data/lm/github/llm-demo/graphrag/ragtest:/workspace/ragtest \
    $IMAGE bash
