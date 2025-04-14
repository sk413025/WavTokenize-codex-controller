import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# import pandas as pd
# import kaldi_io
from sklearn.manifold import TSNE

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Average across sequence dimension
        return self.fc(x)

class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # 確保輸入是2D的 (batch_size, features)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        return self.layer3(x)

class BYOL(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, num_heads, num_layers):
        super().__init__()
        self.online_encoder = SimpleTransformer(input_dim, hidden_dim, num_heads, num_layers)
        self.target_encoder = SimpleTransformer(input_dim, hidden_dim, num_heads, num_layers)
        self.online_projector = ProjectionMLP(hidden_dim, hidden_dim, projection_dim)
        self.target_projector = ProjectionMLP(hidden_dim, hidden_dim, projection_dim)
        self.online_predictor = ProjectionMLP(projection_dim, hidden_dim, projection_dim)

        # Initialize target network
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def target_update(self, tau=0.996):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data
        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    def forward(self, x1, x2):
        online_proj1 = self.online_projector(self.online_encoder(x1))
        online_proj2 = self.online_projector(self.online_encoder(x2))
        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder(x1))
            target_proj2 = self.target_projector(self.target_encoder(x2))

        loss1 = F.mse_loss(online_pred1, target_proj2.detach())
        loss2 = F.mse_loss(online_pred2, target_proj1.detach())
        loss = loss1 + loss2

        return loss

# Training loop
def train_byol(model, optimizer, dataloader, num_epochs, device):
    model.to(device)
    losses = []
    embeddings = []
    all_labels = []
    all_file_paths = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Unpack the batch correctly - now x1 and x2 are already paired
            x1, x2, labels, file_paths = batch
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Calculate BYOL loss directly with the pairs
            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.target_update()
            total_loss += loss.item()

            # Collect embeddings and labels
            with torch.no_grad():
                embeddings.append(model.online_encoder(x1).cpu().numpy())
                all_labels.extend(labels)  # labels are already strings
                all_file_paths.extend(file_paths)

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 保存模型權重
    torch.save(model.state_dict(), './byol_model_weights.pth')
    print("Model weights saved to byol_model_weights.pth")
    
    return losses, np.concatenate(embeddings), all_labels, all_file_paths

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_embeddings(embeddings, labels, title):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps.get_cmap('tab10')  # Updated line

    for i, label in enumerate(unique_labels):
        idx = np.array(labels) == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    label=f'Speaker {label}', color=colors(i), alpha=0.7)

    plt.title(title) # 新增標題
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.show()


def plot_embeddings_3d(embeddings, labels, title):
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps.get_cmap('tab10')  # Updated line

    for i, label in enumerate(unique_labels):
        idx = np.array(labels) == label
        ax.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], reduced_embeddings[idx, 2], label=f'Speaker {label}', alpha=0.7)

    ax.set_title(title) # 新增標題
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()
    plt.show()

# ...existing code...

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 設定要使用的特徵類型和對應的配置
    feature_type = 'encodec'  # 使用encodec特徵
    feature_path = './encodec_features_for_byol.pth'
    input_dim = 256  # encodec特徵維度
    
    print(f"\nUsing {feature_type} features from {feature_path}")
    print(f"Feature dimension: {input_dim}")

    # Hyperparameters
    hidden_dim = 128
    projection_dim = 128
    num_heads = 4
    num_layers = 2
    batch_size = 32
    num_epochs = 7
    learning_rate = 1e-3

    # 加入調試信息
    print("\nModel architecture:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Projection dimension: {projection_dim}")
    
    # Create model with correct input dimension
    model = BYOL(input_dim, hidden_dim, projection_dim, num_heads, num_layers)
    
    # 印出模型結構
    print("\nModel structure:")
    print(model)

    # Create dataset and dataloader with specified feature type
    from byol_dataset import WavFeatureDataset
    print(f"\nLoading dataset from {feature_path}...")
    dataset = WavFeatureDataset(feature_path)
    print(f"Dataset loaded successfully with {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    losses, all_embeddings, all_labels, all_file_paths = train_byol(
        model, optimizer, dataloader, num_epochs, device
    )

    # Plot loss function
    plot_loss(losses)

    # Determine the number of unique speakers
    num_speakers = len(np.unique(all_labels))
    print(f"Number of unique speakers: {num_speakers}")

    # Plot the embeddings without file information
    plot_embeddings(all_embeddings, all_labels, title="2D Embeddings Visualization")

    # Plot the 3D embeddings
    plot_embeddings_3d(all_embeddings, all_labels, title="3D Embeddings Visualization")
