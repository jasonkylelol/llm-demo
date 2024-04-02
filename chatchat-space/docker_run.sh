IMAGE=registry.cn-beijing.aliyuncs.com/chatchat/chatchat:0.2.7-custom-v0.1

debug=1

if [ ${debug} -eq 1 ]; then
    docker run -it --rm --gpus device=0 -h chatchat --name chatchat \
        -p 38501:8501 \
        -v /data/lm/huggingface/models:/data/models \
        --entrypoint 'bash' \
        $IMAGE
else
    docker run -it --rm --gpus device=0 -h chatchat --name chatchat \
        -p 38501:8501 \
        -v /data/lm/huggingface/models:/data/models \
        $IMAGE
fi

# pip3 list | grep nvidia | awk '{print $1}' | xargs pip uninstall -y
# pip3 install --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
