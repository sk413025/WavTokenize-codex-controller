"""
生成 WavTokenizer VQ Codebook Distance Matrix

從 WavTokenizer checkpoint 提取 codebook embeddings，
計算所有 code pairs 之間的 pairwise L2 distance。

輸出: wavtok_distance_mat.pt (4096 x 4096)
"""

import torch
import sys
from pathlib import Path

# 添加 WavTokenizer 路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from decoder.pretrained import WavTokenizer


def compute_codebook_distance_matrix(wavtok_config, wavtok_ckpt, output_path):
    """
    計算 VQ codebook 的 pairwise distance matrix

    Args:
        wavtok_config: WavTokenizer config path
        wavtok_ckpt: WavTokenizer checkpoint path
        output_path: 輸出檔案路徑

    Returns:
        distance_matrix: (K, K) tensor, distance_matrix[i, j] = -||code_i - code_j||²
    """
    print("Loading WavTokenizer...")
    model = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
    model.eval()

    # 取得 VQ codebook
    # WavTokenizer 結構: feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
    quantizer = model.feature_extractor.encodec.quantizer
    codebook = quantizer.vq.layers[0]._codebook.embed  # (K, D), K=4096, D=512

    print(f"Codebook shape: {codebook.shape}")
    K, D = codebook.shape

    # 計算 pairwise distance matrix
    # dist[i, j] = -||code_i - code_j||²
    #            = -(||code_i||² + ||code_j||² - 2 * code_i · code_j)
    print("Computing pairwise distances...")

    # ||code||² for each code
    code_norm_sq = codebook.pow(2).sum(dim=1, keepdim=True)  # (K, 1)

    # Distance matrix
    # distance_matrix[i, j] = -||code_i - code_j||²
    distance_matrix = -(
        code_norm_sq +  # (K, 1) broadcast to (K, K)
        code_norm_sq.t() -  # (1, K) broadcast to (K, K)
        2 * codebook @ codebook.t()  # (K, K)
    )

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    print(f"Distance matrix diagonal (should be ~0): {distance_matrix.diag()[:5]}")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(distance_matrix, output_path)
    print(f"\n✅ Distance matrix saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return distance_matrix


if __name__ == '__main__':
    # 配置
    WAVTOK_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    WAVTOK_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"
    OUTPUT_PATH = "/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/wavtok_distance_mat.pt"

    distance_matrix = compute_codebook_distance_matrix(
        WAVTOK_CONFIG,
        WAVTOK_CKPT,
        OUTPUT_PATH
    )

    print("\n" + "="*70)
    print("Distance matrix generation completed!")
    print("="*70)
