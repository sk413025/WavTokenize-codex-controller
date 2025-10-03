"""
改進的Vector Quantization實現

解決離散化信息損失問題，提高頻譜特徵保留率
主要改進：
1. 使用EMA (Exponential Moving Average) 更新codebook
2. 實現Gumbel Softmax進行軟量化
3. 添加commitment loss和codebook loss
4. 實現multi-scale量化策略

實驗背景：
- 分析發現離散化導致頻譜特徵保留率<70%
- 高頻信息大量丟失
- 需要更sophisticated的量化策略

作者：AI Research Assistant
日期：2025-10-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

class ImprovedVectorQuantizer(nn.Module):
    """改進的向量量化器
    
    基於VQ-VAE和其他先進技術的改進實現
    """
    
    def __init__(
        self,
        num_embeddings: int = 4096,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        use_gumbel: bool = False,
        temperature: float = 1.0
    ):
        """初始化改進的向量量化器
        
        Args:
            num_embeddings: codebook大小
            embedding_dim: 嵌入維度
            commitment_cost: commitment loss權重
            decay: EMA衰減率
            epsilon: 數值穩定性參數
            use_ema: 是否使用EMA更新codebook
            use_gumbel: 是否使用Gumbel Softmax
            temperature: Gumbel Softmax溫度
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        
        # 初始化embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        if use_ema:
            self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
            self._ema_w.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """前向傳播
        
        Args:
            inputs: [batch, ..., embedding_dim] 輸入特徵
            
        Returns:
            quantized: 量化後的特徵
            indices: 量化索引
            loss_dict: 損失字典
        """
        # 保存原始形狀
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        if self.use_gumbel and self.training:
            # 使用Gumbel Softmax進行軟量化
            quantized, indices, loss_dict = self._gumbel_quantize(flat_input)
        else:
            # 使用標準硬量化
            quantized, indices, loss_dict = self._hard_quantize(flat_input)
        
        # 恢復原始形狀
        quantized = quantized.view(input_shape)
        
        return quantized, indices, loss_dict
    
    def _hard_quantize(self, flat_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """硬量化實現"""
        # 計算距離
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # 找到最近的embedding
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        indices_onehot = torch.zeros(indices.shape[0], self.num_embeddings, device=indices.device)
        indices_onehot.scatter_(1, indices, 1)
        
        # 量化
        quantized = torch.matmul(indices_onehot, self.embedding.weight)
        
        # 計算損失
        loss_dict = {}
        
        # Commitment loss - 讓encoder輸出接近chosen embedding
        commitment_loss = F.mse_loss(quantized.detach(), flat_input)
        loss_dict['commitment_loss'] = commitment_loss * self.commitment_cost
        
        if self.use_ema and self.training:
            # 使用EMA更新embedding
            self._update_ema(flat_input, indices_onehot)
            loss_dict['codebook_loss'] = torch.tensor(0.0, device=flat_input.device)
        else:
            # Codebook loss - 讓embedding接近encoder輸出
            codebook_loss = F.mse_loss(quantized, flat_input.detach())
            loss_dict['codebook_loss'] = codebook_loss
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        return quantized, indices.squeeze(1), loss_dict
    
    def _gumbel_quantize(self, flat_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Gumbel Softmax軟量化實現"""
        # 計算距離並轉換為logits
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        logits = -distances / self.temperature
        
        # Gumbel Softmax
        soft_onehot = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        hard_onehot = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        
        # 量化
        soft_quantized = torch.matmul(soft_onehot, self.embedding.weight)
        hard_quantized = torch.matmul(hard_onehot, self.embedding.weight)
        
        # 獲取hard indices用於記錄
        indices = torch.argmax(hard_onehot, dim=1)
        
        # 使用soft量化進行前向傳播，但在反向傳播中使用hard
        quantized = soft_quantized + (hard_quantized - soft_quantized).detach()
        
        # 計算損失
        loss_dict = {}
        commitment_loss = F.mse_loss(quantized.detach(), flat_input)
        loss_dict['commitment_loss'] = commitment_loss * self.commitment_cost
        loss_dict['codebook_loss'] = torch.tensor(0.0, device=flat_input.device)
        
        return quantized, indices, loss_dict
    
    def _update_ema(self, flat_input: torch.Tensor, indices_onehot: torch.Tensor):
        """使用EMA更新embedding weights"""
        # 更新cluster sizes
        cluster_size = torch.sum(indices_onehot, dim=0)
        self._ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        
        # 更新embedding weights
        dw = torch.matmul(indices_onehot.t(), flat_input)
        self._ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # 正規化
        n = torch.sum(self._ema_cluster_size)
        cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        
        normalized_w = self._ema_w / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(normalized_w)

class MultiScaleVectorQuantizer(nn.Module):
    """多尺度向量量化器
    
    使用不同尺度的codebook捕獲不同層次的特徵
    """
    
    def __init__(
        self,
        scales: list = [4096, 1024, 256],
        embedding_dim: int = 512,
        **kwargs
    ):
        """初始化多尺度量化器
        
        Args:
            scales: 不同尺度的codebook大小
            embedding_dim: 嵌入維度
            **kwargs: 其他VectorQuantizer參數
        """
        super().__init__()
        
        self.scales = scales
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(
                num_embeddings=scale,
                embedding_dim=embedding_dim,
                **kwargs
            ) for scale in scales
        ])
        
        # 學習各尺度的權重
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, list, dict]:
        """前向傳播
        
        Returns:
            weighted_quantized: 加權量化結果
            all_indices: 所有尺度的索引
            combined_loss_dict: 合併的損失字典
        """
        quantized_results = []
        all_indices = []
        combined_loss_dict = {}
        
        # 應用各尺度量化
        for i, quantizer in enumerate(self.quantizers):
            quantized, indices, loss_dict = quantizer(inputs)
            quantized_results.append(quantized)
            all_indices.append(indices)
            
            # 合併損失
            for key, value in loss_dict.items():
                if key not in combined_loss_dict:
                    combined_loss_dict[key] = 0.0
                combined_loss_dict[key] += value * self.scale_weights[i]
        
        # 加權合併量化結果
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        weighted_quantized = sum(
            w * q for w, q in zip(scale_weights_norm, quantized_results)
        )
        
        return weighted_quantized, all_indices, combined_loss_dict

def create_wavtokenizer_vq_wrapper(
    wavtokenizer_model,
    use_improved_vq: bool = True,
    use_multiscale: bool = False,
    **vq_kwargs
):
    """創建WavTokenizer的改進VQ包裝器
    
    Args:
        wavtokenizer_model: 原始WavTokenizer模型
        use_improved_vq: 是否使用改進的VQ
        use_multiscale: 是否使用多尺度VQ
        **vq_kwargs: VQ參數
        
    Returns:
        包裝後的模型
    """
    class WavTokenizerVQWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.wavtokenizer = wavtokenizer_model
            
            # 替換quantizer
            if use_multiscale:
                self.improved_quantizer = MultiScaleVectorQuantizer(**vq_kwargs)
            elif use_improved_vq:
                self.improved_quantizer = ImprovedVectorQuantizer(**vq_kwargs)
            else:
                self.improved_quantizer = None
        
        def encode_infer(self, audio, bandwidth_id=None):
            """編碼推理，使用改進的量化器"""
            # 使用原始encoder獲取特徵
            with torch.no_grad():
                # 這裡需要根據實際WavTokenizer API調整
                features = self.wavtokenizer.encode(audio, bandwidth_id)
                
                if self.improved_quantizer is not None:
                    # 使用改進的量化器
                    if isinstance(features, list):
                        # 處理多層特徵
                        quantized_features = []
                        for feat in features:
                            if isinstance(feat, list):
                                layer_features = []
                                for layer_feat in feat:
                                    q_feat, _, _ = self.improved_quantizer(layer_feat)
                                    layer_features.append(q_feat)
                                quantized_features.append(layer_features)
                            else:
                                q_feat, _, _ = self.improved_quantizer(feat)
                                quantized_features.append(q_feat)
                        return quantized_features
                    else:
                        quantized, _, _ = self.improved_quantizer(features)
                        return quantized
                else:
                    return features
        
        def decode(self, features, bandwidth_id=None):
            """解碼，直接使用原始decoder"""
            return self.wavtokenizer.decode(features, bandwidth_id)
        
        def forward(self, audio, bandwidth_id=None):
            """完整的編碼-解碼過程"""
            encoded = self.encode_infer(audio, bandwidth_id)
            decoded = self.decode(encoded, bandwidth_id)
            return decoded
    
    return WavTokenizerVQWrapper()

if __name__ == "__main__":
    print("🔧 改進的Vector Quantization實現完成！")
    print("\n主要改進：")
    print("1. EMA更新策略 - 穩定codebook學習")
    print("2. Gumbel Softmax - 可微分軟量化")
    print("3. 多尺度量化 - 捕獲不同層次特徵")
    print("4. 改進的損失函數 - 更好的量化品質")
    print("\n預期效果：")
    print("- 提高頻譜特徵保留率從<70%到>85%")
    print("- 減少高頻信息損失")
    print("- 改善音頻重建質量")
    print("- 更穩定的訓練過程")