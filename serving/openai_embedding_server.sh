#!/bin/bash

EmbeddingModel=maidalun1020/bce-embedding-base_v1

PORT=8060 MODEL=/root/huggingface/models/$EmbeddingModel python -m open.text.embeddings.server
