import json
from pathlib import Path
from datetime import datetime
import torch

# Paths from exp_1201 config
TRAIN_CACHE = Path('/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt')
VAL_CACHE = Path('/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt')

OUT_DIR = Path(__file__).resolve().parent


def load_cache(path: Path):
    return torch.load(path, map_location='cpu')


def extract_sets(records):
    # Expected keys from cache
    keys = [
        'content_id',
        'speaker_id',
        'sentence_id',
        'filename',
        'noisy_path',
        'clean_path',
        'material',
    ]

    sets = {k: set() for k in keys}
    pair_sets = {
        'noisy_clean_path': set(),
        'speaker_sentence': set(),
        'speaker_content': set(),
    }

    for item in records:
        for k in keys:
            if k in item and item[k] is not None:
                sets[k].add(item[k])

        if 'noisy_path' in item and 'clean_path' in item:
            pair_sets['noisy_clean_path'].add((item['noisy_path'], item['clean_path']))
        if 'speaker_id' in item and 'sentence_id' in item:
            pair_sets['speaker_sentence'].add((item['speaker_id'], item['sentence_id']))
        if 'speaker_id' in item and 'content_id' in item:
            pair_sets['speaker_content'].add((item['speaker_id'], item['content_id']))

    return sets, pair_sets


def intersect_stats(train_set, val_set):
    inter = train_set & val_set
    return {
        'train_count': len(train_set),
        'val_count': len(val_set),
        'intersection': len(inter),
        'val_overlap_ratio': (len(inter) / len(val_set)) if val_set else 0.0,
        'train_overlap_ratio': (len(inter) / len(train_set)) if train_set else 0.0,
    }


def main():
    train = load_cache(TRAIN_CACHE)
    val = load_cache(VAL_CACHE)

    train_sets, train_pairs = extract_sets(train)
    val_sets, val_pairs = extract_sets(val)

    stats = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'train_cache': str(TRAIN_CACHE),
        'val_cache': str(VAL_CACHE),
        'train_len': len(train),
        'val_len': len(val),
        'keys': {},
        'pairs': {},
    }

    for k in train_sets:
        stats['keys'][k] = intersect_stats(train_sets[k], val_sets[k])

    for k in train_pairs:
        stats['pairs'][k] = intersect_stats(train_pairs[k], val_pairs[k])

    out_json = OUT_DIR / 'cache_overlap_stats.json'
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # Simple markdown summary
    lines = [
        '# Cache Overlap Check (H6)',
        '',
        f'- timestamp: {stats["timestamp"]}',
        f'- train_len: {stats["train_len"]}',
        f'- val_len: {stats["val_len"]}',
        '',
        '## Key overlaps',
    ]

    for k, v in stats['keys'].items():
        lines.append(
            f'- {k}: inter={v["intersection"]} (val_overlap={v["val_overlap_ratio"]:.4f}, train_overlap={v["train_overlap_ratio"]:.4f})'
        )

    lines.append('')
    lines.append('## Pair overlaps')
    for k, v in stats['pairs'].items():
        lines.append(
            f'- {k}: inter={v["intersection"]} (val_overlap={v["val_overlap_ratio"]:.4f}, train_overlap={v["train_overlap_ratio"]:.4f})'
        )

    out_md = OUT_DIR / 'cache_overlap_report.md'
    out_md.write_text('\n'.join(lines) + '\n')

    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_md}')


if __name__ == '__main__':
    main()
