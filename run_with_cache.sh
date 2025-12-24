#!/bin/bash

# Example workflow for using cached embeddings

# Step 1: Extract embeddings (run once)
echo "Step 1: Extracting embeddings..."
CUDA_VISIBLE_DEVICES=1 nohup python extract_embeddings.py \
  --data-csv datasets/sampled.csv \
  --output-dir cache/embeddings \
  --batch-size 4 \
  --device cuda > extract_embeddings.log 2>&1 &



echo "Done!"
# Step 2: Train using cached embeddings (can run many times with different hyperparameters)
# echo ""
# echo "Step 2: Training with cached embeddings..."
# python main.py \
#   --cached-embeddings-dir cache/embeddings \
#   --tuning-mode head \
#   --hidden-dim 512 \
#   --mlp-hidden 256 \
#   --batch-size 8 \
#   --epochs 10 \
#   --lr 1e-4 \
#   --device cuda \
#   --fusion-mode concat

# echo ""
# echo "Done! You can now re-run training with different hyperparameters without re-extracting embeddings."
