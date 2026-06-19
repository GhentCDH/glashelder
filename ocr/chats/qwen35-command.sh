docker run -d \
  --name qwen35-miel \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/bavercru/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly   Qwen/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'
