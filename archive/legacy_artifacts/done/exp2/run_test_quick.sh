#!/bin/bash

echo "============================================"
echo "EXP2 最小化測試 (正確的 WavTokenizer 路徑)"
echo "============================================"

cd /home/sbplab/ruizi/c_code

python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/test_minimal \
    --wavtokenizer_config config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --wavtokenizer_checkpoint models/wavtokenizer_large_speech_320_24k.ckpt \
    --lambda_speaker 0.5 \
    --speaker_model_type ecapa \
    --num_epochs 10 \
    --batch_size 14 \
    --max_sentences_per_speaker 1 \
    --learning_rate 1e-4 \
    --num_layers 2
