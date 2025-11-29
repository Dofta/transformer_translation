import os
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# --- 配置 ---
# 指向包含那些 arrow 文件的目录，或者 dataset 的根目录
# 根据你的描述，应该是 'data/dataset' 或者 'data/dataset/Mxode__bi_st'
# 如果 load_from_disk 报错，可以尝试把 path 指向 arrow 文件所在的上一级
RAW_DATA_PATH = "./data/dataset" 

OUTPUT_DIR = "./data"
SANITY_SIZE = 20000
MAIN_TRAIN_SIZE = 1500000 
TEST_SIZE = 5000

MIN_LEN = 4
MAX_LEN = 100

def save_to_text(dataset, output_subdir, split_name):
    """将 HF dataset 对象导出为纯文本文件"""
    dir_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(dir_path, exist_ok=True)
    
    cn_path = os.path.join(dir_path, f"{split_name}.cn")
    en_path = os.path.join(dir_path, f"{split_name}.en")
    
    print(f"Saving {len(dataset)} examples to {dir_path}...")
    
    with open(cn_path, 'w', encoding='utf-8') as f_cn, \
         open(en_path, 'w', encoding='utf-8') as f_en:
        
        # 假设列名是 'zh' 和 'en'，如果不确定，脚本运行通过 print(ds.column_names) 确认
        # BiST 通常是 src, tgt 或者 zh, en
        src_col = 'zh' if 'zh' in dataset.column_names else 'src'
        tgt_col = 'en' if 'en' in dataset.column_names else 'tgt'
        
        for item in tqdm(dataset):
            # 去除换行符，保证一行一句
            src_text = item[src_col].replace('\n', ' ').strip()
            tgt_text = item[tgt_col].replace('\n', ' ').strip()
            
            if src_text and tgt_text:
                f_cn.write(src_text + '\n')
                f_en.write(tgt_text + '\n')

def filter_fn(example):
    """过滤函数"""
    # 假设列名是 zh 和 en，需要根据实际 print 结果调整
    src = example.get('zh') or example.get('src') or example.get('source')
    tgt = example.get('en') or example.get('tgt') or example.get('target')
    
    if not src or not tgt:
        return False
    if len(src) < MIN_LEN or len(tgt) < MIN_LEN:
        return False
    # 简单估算长度，避免 tokenizer 可以在这里省时间
    if len(src) > MAX_LEN or len(tgt.split()) > MAX_LEN:
        return False
    return True

def main():
    print("Loading dataset from disk...")
    try:
        # 尝试方法 1: 直接加载本地 Arrow 格式
        # 如果你的目录结构是标准的 HF save_to_disk 结构，这行代码能工作
        ds = load_from_disk(RAW_DATA_PATH)
    except Exception as e:
        print(f"load_from_disk failed: {e}")
        print("Trying generic loading script...")
        # 尝试方法 2: 如果只是包含 parquet/arrow 的文件夹
        ds = load_dataset(RAW_DATA_PATH, split='train')

    if isinstance(ds, dict): # 如果加载出来是 DatasetDict (包含 train/test keys)
        ds = ds['train']

    print(f"Original Size: {len(ds)}")
    print(f"Column Names: {ds.column_names}") # 重要：看一眼列名！

    # --- 1. 过滤 ---
    print("Filtering dataset...")
    # num_proc=8 利用你的多核 CPU 加速过滤
    ds_filtered = ds.filter(filter_fn, num_proc=8) 
    print(f"Filtered Size: {len(ds_filtered)}")

    # --- 2. 打乱 ---
    print("Shuffling...")
    ds_shuffled = ds_filtered.shuffle(seed=42)

    # --- 3. 切分与导出 ---
    
    # A. Sanity (20k)
    ds_sanity = ds_shuffled.select(range(SANITY_SIZE))
    save_to_text(ds_sanity, "sanity", "train")
    
    # B. Test (5k) - 取最后的 5k
    total_len = len(ds_shuffled)
    ds_test = ds_shuffled.select(range(total_len - TEST_SIZE, total_len))
    save_to_text(ds_test, "test", "test")
    
    # C. Production (1.5M)
    # 取 Sanity 之后的数据
    remain_start_idx = SANITY_SIZE
    remain_end_idx = remain_start_idx + MAIN_TRAIN_SIZE
    
    if remain_end_idx > total_len - TEST_SIZE:
        print("Warning: Not enough data for requested 1.5M, using available.")
        remain_end_idx = total_len - TEST_SIZE

    ds_prod = ds_shuffled.select(range(remain_start_idx, remain_end_idx))
    save_to_text(ds_prod, "bist_1.5m", "train")

    print("All Done! Data is ready in './data/' folder.")

if __name__ == "__main__":
    main()