```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
cd source
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

uv venv -p 3.12
source .venv/bin/activate
python scripts/prepare_data.py --dataset sharegpt
echo $CUDA_HOME

torchrun \
    --standalone \
    --nproc_per_node gpu \
    scripts/train_eagle3.py \
    --target-model-path /root/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/8f726c6a497fd439f0d6f726e52f8e3b439a26e5 \
    --draft-model-config configs/gemma3-270m-eagle3.json \
    --train-data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-dir ./outputs/gemma3-270m-eagle3 \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template gemma \
    --target-model-backend sglang \
```
