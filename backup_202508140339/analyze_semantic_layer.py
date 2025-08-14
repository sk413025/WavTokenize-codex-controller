"""
分析各 residual block 層語意表徵強度（t-SNE 可視化）
自動儲存每層 t-SNE 圖，並在檔名中標註日期、實驗編號與層數。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime

# 載入模型與資料（請根據實際情況修改）


# 假設你有一個 dataloader
# from your_dataset import get_dataloader
# dataloader = get_dataloader(...)

def pad_collate_fn(batch):
    """
    將 batch 內的音檔自動 zero-padding 到同一長度，回傳 padded tensor 及其他欄位。
    Args:
        batch: List of tuples, 每個元素通常為 (waveform, ...)
    Returns:
        Tuple: (padded_waveforms, ...)
    """
    import torch
    # 只 pad 第一個欄位（waveform），其他欄位直接組成 list
    waveforms = [item[0] for item in batch]
    max_len = max(w.shape[1] for w in waveforms)
    padded = []
    for w in waveforms:
        pad_len = max_len - w.shape[1]
        if pad_len > 0:
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded.append(w)
    padded_waveforms = torch.stack(padded)
    # 若有其他欄位，組成 list of list
    others = list(zip(*batch))
    if len(others) > 1:
        return (padded_waveforms,) + tuple([list(x) for x in others[1:]])
    else:
        return (padded_waveforms,)

def extract_intermediate_features(model, dataloader, device):
    """
    從 dataloader 取一個 batch，回傳所有 residual block 層的特徵
    Returns: List[np.ndarray]，每層 shape: [N, C, T]
    """
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_wav = batch[0].to(device)
            # 只取一個 batch
            output_tuple = model(input_wav)
            # output_tuple: (output, input_features, enhanced_features, discrete_code, intermediate_enhanced_features, intermediate_features_list)
            intermediate_features_list = output_tuple[5]
            # 轉 numpy
            features_np = [f.detach().cpu().numpy() for f in intermediate_features_list]
            return features_np
    return None

def plot_tsne_for_layers(features_list, save_dir, exp_id="exp1"):
    """
    對每一層做 t-SNE，並儲存圖檔
    """
    os.makedirs(save_dir, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    for i, feat in enumerate(features_list):
        # [N, C, T] → [N, C] (mean pooling)
        if feat.ndim == 3:
            feat_pooled = np.mean(feat, axis=2)
        else:
            feat_pooled = feat
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(feat_pooled)
        plt.figure(figsize=(8, 7))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, s=40)
        plt.title(f"t-SNE Layer {i+1} ({exp_id}, {date_str})", fontsize=16)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"tsne_{exp_id}_layer{i+1}_{date_str}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Layer {i+1} t-SNE saved: {save_path}")

if __name__ == "__main__":
    # === 自動載入模型與資料集（參考 ttt2.py） ===
    from try3 import AudioDataset  # 修正：從 try3 匯入 AudioDataset
    from ttt2 import EnhancedWavTokenizer
    import yaml

    # 設定 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 指定 config 與訓練後最佳權重路徑
    config_path = os.path.join(os.getcwd(), "config", "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    # 使用訓練後最佳權重
    model_path = os.path.join(os.getcwd(), "results", "tsne_outputs", "output4_202507280545", "best_model.pth")

    # 載入模型
    model = EnhancedWavTokenizer(config_path, model_path).to(device)
    model.eval()

    # 指定資料路徑（依 ttt2.py）
    input_dirs = os.path.join(os.getcwd(), "data", "raw", "box")
       
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")

    # 載入資料集
    dataset = AudioDataset(input_dirs=input_dirs, target_dir=target_dir, max_files_per_dir=20)  # 可調整 max_files_per_dir
    # 為避免 CUDA OOM，建議 batch_size=1，如需加速可再調整
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)

    # 執行特徵提取與 t-SNE 分析
    features_list = extract_intermediate_features(model, dataloader, device)
    plot_tsne_for_layers(features_list, save_dir="semantic_layer_tsne", exp_id="exp1")
