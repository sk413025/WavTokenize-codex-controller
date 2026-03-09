"""
Shared configuration constants for audio denoising experiments

使用說明:
    from config import PAD_TOKEN, CODEBOOK_SIZE, SAMPLE_RATE
"""

# Padding token for sequence alignment
# Must be outside codebook range [0, CODEBOOK_SIZE-1]
PAD_TOKEN = 4096

# WavTokenizer codebook size
# Valid token range: [0, CODEBOOK_SIZE-1]
CODEBOOK_SIZE = 4096

# Audio sample rate (Hz)
SAMPLE_RATE = 24000

# Speaker embedding dimension
SPEAKER_EMBED_DIM = 256
