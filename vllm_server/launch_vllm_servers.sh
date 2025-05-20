#!/usr/bin/zsh
source ~/.zshrc

MODEL="Qwen/Qwen3-4B"
HOST="0.0.0.0"
PORT_BASE=10000
CONTEXT_LEN=32768
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="logs"
mkdir -p $LOGS_DIR


# ====================================

# (re-)launch nginx
NGINX_LOG_FILE="${LOGS_DIR}/${CURRENT_TIME}_nginx.log"
/iris/u/yoonho/nginx/sbin/nginx -s stop
/iris/u/yoonho/nginx/sbin/nginx -c $(pwd)/nginx.conf >> $NGINX_LOG_FILE 2>&1
echo "Nginx is running!"
lsof -i :8000

# ====================================

export TOKENIZERS_PARALLELISM=false
if [ -d "/scr-ssd" ]; then
    export VLLM_CACHE_ROOT=/scr-ssd/yoonho/vllm_cache
else
    export VLLM_CACHE_ROOT=/scr/yoonho/vllm_cache
fi
export TORCH_CUDA_ARCH_LIST="9.0;8.9+PTX"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_PRECOMPILED=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "Killing any existing GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9
for GPU in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((PORT_BASE + GPU))
  echo "Killing any existing processes on port $PORT..."
  lsof -tiTCP:$PORT -sTCP:LISTEN | xargs -r kill -9

  echo "Starting vLLM server on GPU $GPU at port $PORT"
  CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --host $HOST --port $PORT \
    --model $MODEL \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len $CONTEXT_LEN --max-seq-len-to-capture $CONTEXT_LEN \
    | tee "${LOGS_DIR}/${CURRENT_TIME}_vllm_${GPU}.log" 2>&1 &
done
echo "vLLM servers are running!"
echo "Access the API through port 8000" 
wait

cleanup() {
    echo "Received termination signal. Cleaning up..."
    
    echo "Stopping vLLM servers..."
    pkill -f "python -m vllm.entrypoints.openai.api_server"
    
    echo "Stopping GPU processes..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9

    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        PORT=$((PORT_BASE + GPU))
        echo "Killing any existing processes on port $PORT..."
        lsof -tiTCP:$PORT -sTCP:LISTEN | xargs -r kill -9
    done

    echo "Stopping nginx..."
    /iris/u/yoonho/nginx/sbin/nginx -s stop
    
    echo "Cleanup complete. Exiting."
    exit 0
}

trap cleanup SIGINT SIGTERM
