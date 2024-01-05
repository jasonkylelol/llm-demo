
model=/models/HuggingFaceH4/zephyr-7b-beta/
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
models_dir=$PWD/models

docker run --gpus all --shm-size 1g -p 8080:80 \
	-v $volume:/data \
	-v $models_dir:/models \
	ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --quantize bitsandbytes-fp4

