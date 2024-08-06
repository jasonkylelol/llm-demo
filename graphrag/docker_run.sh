IMAGE=graphrag:dev-0.1

docker run -it --rm --gpus all -h graph-rag --name graph-rag \
    -v /data/lm/github/llm-demo/graphrag:/workspace/dev \
    $IMAGE bash
