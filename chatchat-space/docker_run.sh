IMAGE=registry.cn-beijing.aliyuncs.com/chatchat/chatchat:0.2.7-custom-v0.1

docker run -it --rm --gpus device=0 -h chatchat --name chatchat \
    -p 38051:8501 \
    --entrypoint 'bash' \
    $IMAGE

# docker run -it --rm -d --gpus device=0 -h chatchat --name chatchat \
    # -p 80:8501 $IMAGE bash