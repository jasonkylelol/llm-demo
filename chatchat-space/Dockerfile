from registry.cn-beijing.aliyuncs.com/chatchat/chatchat:0.2.7
run apt update; apt install -y ca-certificates curl wget vim net-tools
run apt reinstall -y bash bash-completion

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple

run rm -rf /chatglm2-6b
run rm -rf /text2vec-bge-large-chinese
run rm -f /zh_core_web_lg-3.6.0-py3-none-any.whl

add configs/model_config.py /langchain-chatchat/configs/model_config.py
add configs/server_config.py /langchain-chatchat/configs/server_config.py

entrypoint ["bash", "-il", "start_service.sh"]
