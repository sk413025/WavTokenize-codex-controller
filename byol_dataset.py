import torch
from torch.utils.data import Dataset
import kaldiio
import numpy as np
import re
from typing import Tuple, Dict, List

class XVectorDataset(Dataset):
    """Dataset for loading x-vectors with different materials for BYOL training"""
    
    def __init__(self, scp_path: str):
        """
        Args:
            scp_path: Path to xvector.scp file
        """
        self.xvector_dict = kaldiio.load_scp(scp_path)
        self.keys = list(self.xvector_dict.keys())

        # Debugging: Print the number of loaded x-vectors
        print(f"Loaded {len(self.keys)} x-vectors")
        print(f"Keys: {self.keys}")

        # Group utterances by speaker and material
        self.spk_mat_dict = self._group_utterances()
        
        # Create pairs list for training
        self.pairs = self._create_pairs()

    def _group_utterances(self) -> Dict[str, Dict[str, List[str]]]:
        """Group utterances by speaker and material"""
        spk_mat_dict = {}
        
        for key in self.keys:
            # Parse key like 'boy1_box_LDV_001'
            parts = key.split('_')
            if len(parts) < 3:
                continue
                
            spk = parts[0]
            material = parts[1]
            
            if spk not in spk_mat_dict:
                spk_mat_dict[spk] = {}
            
            if material not in spk_mat_dict[spk]:
                spk_mat_dict[spk][material] = []
                
            spk_mat_dict[spk][material].append(key)
            
        return spk_mat_dict

    def _create_pairs(self) -> List[Tuple[str, str]]:
        """Create pairs of utterances for training, including clean utterances"""
        pairs = []
        materials = ['box', 'papercup', 'plastic']
        
        for spk in self.spk_mat_dict:
            # 1. 先配對材質與對應的 clean
            for mat in materials:
                if (mat not in self.spk_mat_dict[spk]):
                    continue
                    
                for utt1 in self.spk_mat_dict[spk][mat]:
                    # Extract utterance number
                    utt_num = re.search(r'(\d+)$', utt1).group(1)
                    
                    # Find matching clean utterance
                    clean_candidates = [u for u in self.spk_mat_dict[spk].get(mat + '_clean', [])
                                     if u.endswith(utt_num)]
                    if clean_candidates:
                        pairs.append((utt1, clean_candidates[0]))
            
            # 2. 再配對不同材質之間
            for i, mat1 in enumerate(materials):
                if mat1 not in self.spk_mat_dict[spk]:
                    continue
                    
                for utt1 in self.spk_mat_dict[spk][mat1]:
                    utt_num = re.search(r'(\d+)$', utt1).group(1)
                    
                    # Only pair with materials after mat1 to avoid duplicates
                    for mat2 in materials[i+1:]:
                        if mat2 not in self.spk_mat_dict[spk]:
                            continue
                            
                        utt2_candidates = [u for u in self.spk_mat_dict[spk][mat2] 
                                        if u.endswith(utt_num)]
                        if utt2_candidates:
                            pairs.append((utt1, utt2_candidates[0]))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        utt1, utt2 = self.pairs[idx]
        
        # Load x-vectors and make writable copies
        x1 = torch.from_numpy(np.array(self.xvector_dict[utt1], copy=True)).float()
        x2 = torch.from_numpy(np.array(self.xvector_dict[utt2], copy=True)).float()
        
        # Extract label (assuming speaker ID is the first part of the key)
        label = utt1.split('_')[0]  # Assuming label is a string
        
        return x1, x2, label, utt1  # Return the file path as well
        
    @property
    def feat_dim(self) -> int:
        """Return feature dimension"""
        return next(iter(self.xvector_dict.values())).shape[0]
'''
class WavFeatureDataset(Dataset):
    def __init__(self, features_path):
        # 載入特徵文件
        data = torch.load(features_path)
        self.features = data['features']  # 應該已經是正確的形狀 (batch_size, seq_len, input_dim)
        self.labels = data['labels']
        self.file_paths = data['file_paths']
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]  # 獲取形狀為 (seq_len, input_dim) 的特徵
        
        # 對每個時間步長進行數據增強
        noise1 = torch.randn_like(feature) * 0.05
        noise2 = torch.randn_like(feature) * 0.05
        aug1 = feature + noise1
        aug2 = feature + noise2
        
        return aug1, aug2, self.labels[idx], self.file_paths[idx]
'''

class WavFeatureDataset(Dataset):
    def __init__(self, feature_path):
        self.data = torch.load(feature_path)  # 載入 .pth 檔案
        self.features = self.data['features']
        self.labels = self.data['labels']
        self.file_paths = self.data['file_paths']
        
        # 創建視圖1和視圖2（這裡簡化為對原始特徵進行微小的擾動）
        self.view1 = self.features + torch.randn_like(self.features) * 0.01
        self.view2 = self.features + torch.randn_like(self.features) * 0.01

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回配對的視圖以及標籤和文件名
        return self.view1[idx], self.view2[idx], self.labels[idx], self.file_paths[idx]

if __name__ == '__main__':
    # 示例使用
    feature_path = './encodec_features_for_byol.pth'
    dataset = WavFeatureDataset(feature_path)
    print(f"Dataset size: {len(dataset)}")
    
    # 獲取一個樣本
    sample1, sample2, label, file_path = dataset[0]
    print(f"Sample 1 shape: {sample1.shape}")
    print(f"Sample 2 shape: {sample2.shape}")
    print(f"Label: {label}")
    print(f"File path: {file_path}")