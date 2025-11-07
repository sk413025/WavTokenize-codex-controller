import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
from decoder.pretrained import WavTokenizer


def setup_logger(output_dir):
    log_file = Path(output_dir) / 'training.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        logits = model(noisy, spk, return_logits=True)
        B,T,V = logits.shape
        loss = criterion(logits.view(B*T, V), clean.view(B*T))
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == clean).sum().item()
        total_tokens += B*T
    return {'loss': total_loss/len(dataloader), 'accuracy': total_correct/total_tokens*100}


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for batch in dataloader:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)
        logits = model(noisy, spk, return_logits=True)
        B,T,V = logits.shape
        loss = criterion(logits.view(B*T, V), clean.view(B*T))
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == clean).sum().item()
        total_tokens += B*T
    return {'loss': total_loss/len(dataloader), 'accuracy': total_correct/total_tokens*100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', default='./data')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--dim_feedforward', type=int, default=2048)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--speaker_tokens', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_epochs', type=int, default=10)
    ap.add_argument('--learning_rate', type=float, default=1e-4)
    args = ap.parse_args()

    # env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dirs
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/crossattn_k4_gate_{args.num_epochs}ep_{timestamp}'
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)
    logger.info('Cross-Attn Gated Experiment (small run)')

    # cache
    cache_dir = Path(args.cache_dir)
    train_ds = ZeroShotAudioDatasetCached(str(cache_dir/'train_cache.pt'))
    val_ds = ZeroShotAudioDatasetCached(str(cache_dir/'val_cache.pt'))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=cached_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=cached_collate_fn, pin_memory=True)

    # wavtokenizer to get codebook
    # Keep WavTokenizer on CPU to save GPU memory; extract codebook then free
    wavtokenizer = WavTokenizer.from_pretrained0802(
        'config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
        'models/wavtokenizer_large_speech_320_24k.ckpt'
    )
    wavtokenizer.eval()
    with torch.no_grad():
        codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook.detach().cpu()
    try:
        del wavtokenizer
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # model/opt
    model = ZeroShotDenoisingTransformerCrossAttnGated(
        codebook=codebook, speaker_embed_dim=256, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        speaker_tokens=args.speaker_tokens
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    # save config
    with open(outdir/'config.json','w') as f: json.dump(vars(args), f, indent=2)

    best_val = 0.0
    for epoch in range(1, args.num_epochs+1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = validate_epoch(model, val_loader, criterion, device)
        logger.info(f'Epoch {epoch}/{args.num_epochs}')
        logger.info(f'  Train - Loss: {tr["loss"]:.4f}, Acc: {tr["accuracy"]:.2f}%')
        logger.info(f'  Val   - Loss: {va["loss"]:.4f}, Acc: {va["accuracy"]:.2f}%')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, outdir/f'checkpoint_epoch_{epoch}.pth')
        if va['accuracy'] > best_val:
            best_val = va['accuracy']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, outdir/'best_model.pth')
            logger.info(f'  ✓ 保存最佳模型 (Val Acc: {best_val:.2f}%)')

if __name__ == '__main__':
    main()
