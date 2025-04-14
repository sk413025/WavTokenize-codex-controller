import os

def generate_file_lists(source_dir, target_dir, output_dir):
    """生成訓練所需的文件列表
    
    Args:
        source_dir: 源音頻目錄路徑 (例如: ./data/train/source)
        target_dir: 目標音頻目錄路徑 (例如: ./data/train/target)  
        output_dir: 輸出文件目錄路徑 (例如: ./data/train)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取源文件列表
    source_files = []
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith('.wav'):
            # 使用相對路徑,方便遷移
            source_path = os.path.join('./data/train/source', filename)
            source_files.append(source_path)
    
    # 生成source_list.txt
    with open(os.path.join(output_dir, "source_list.txt"), "w", encoding="utf-8") as f:
        for source_path in source_files:
            f.write(f"{source_path}\n")
            
    # 生成speaker_pairs.txt
    with open(os.path.join(output_dir, "speaker_pairs.txt"), "w", encoding="utf-8") as f:
        for source_path in source_files:
            # 從源文件名推導目標文件名
            basename = os.path.basename(source_path)
            target_basename = basename.replace('box_LDV', 'box_clean')
            target_path = os.path.join('./data/train/target', target_basename)
            
            # 確認目標文件存在
            abs_target_path = os.path.join(target_dir, target_basename)
            if os.path.exists(abs_target_path):
                f.write(f"{source_path}|{target_path}\n")
            else:
                print(f"Warning: No matching target file {target_path}")

if __name__ == "__main__":
    data_root = "./data"
    source_dir = os.path.join(data_root, "train/source")
    target_dir = os.path.join(data_root, "train/target") 
    output_dir = os.path.join(data_root, "train")
    
    generate_file_lists(source_dir, target_dir, output_dir)
    
    # 打印示例和統計
    with open(os.path.join(output_dir, "source_list.txt")) as f:
        source_lines = f.readlines()
        print("\nExample source_list.txt:")
        print(source_lines[0].strip())
        print(f"Total {len(source_lines)} source files")
        
    with open(os.path.join(output_dir, "speaker_pairs.txt")) as f:
        pair_lines = f.readlines()
        print("\nExample speaker_pairs.txt:")
        print(pair_lines[0].strip())
        print(f"Total {len(pair_lines)} pairs")
