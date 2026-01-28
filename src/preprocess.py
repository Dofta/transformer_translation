import os
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_PATH = "./data/dataset"
OUTPUT_DIR = "./data"

SANITY_SIZE = 100000
MAIN_TRAIN_SIZE = 1500000
TEST_SIZE = 5000

MIN_LEN = 4
MAX_LEN = 64


def save_to_text(dataset, output_subdir, split_name):
    """Export HF dataset object to plain text files"""
    dir_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(dir_path, exist_ok=True)

    cn_path = os.path.join(dir_path, f"{split_name}.cn")
    en_path = os.path.join(dir_path, f"{split_name}.en")

    print(f"Saving {len(dataset)} examples to {dir_path}...")

    with open(cn_path, "w", encoding="utf-8") as f_cn, open(
        en_path, "w", encoding="utf-8"
    ) as f_en:

        for item in tqdm(dataset):
            src_text = item.get("text_zh")
            tgt_text = item.get("text_en")

            if src_text and tgt_text:
                # Remove newline characters to ensure one sentence per line
                src_clean = str(src_text).replace("\n", " ").strip()
                tgt_clean = str(tgt_text).replace("\n", " ").strip()

                if src_clean and tgt_clean:
                    f_cn.write(src_clean + "\n")
                    f_en.write(tgt_clean + "\n")


def filter_fn(example):
    """Filter function"""
    src = example.get("text_zh")
    tgt = example.get("text_en")

    if not src or not tgt:
        return False

    if len(src) < MIN_LEN or len(tgt) < MIN_LEN:
        return False
    if len(src) > MAX_LEN or len(tgt.split()) > MAX_LEN:
        return False

    return True


def main():
    print("Loading dataset...")
    try:
        ds = load_from_disk(RAW_DATA_PATH)
    except:
        ds = load_dataset(RAW_DATA_PATH, split="train")

    if isinstance(ds, dict) and "train" in ds:
        ds = ds["train"]

    print(f"Original Size: {len(ds)}")
    print(f"Columns found: {ds.column_names}")

    # --- Filtering ---
    print("Filtering dataset...")
    ds_filtered = ds.filter(filter_fn, num_proc=8)
    print(f"Filtered Size: {len(ds_filtered)}")

    if len(ds_filtered) == 0:
        raise ValueError(
            "Error: Filtered dataset is empty! Please check filter_fn logic or MAX_LEN."
        )

    # --- Shuffling ---
    print("Shuffling...")
    ds_shuffled = ds_filtered.shuffle(seed=42)

    # --- Splitting and Exporting ---
    total_len = len(ds_shuffled)

    # A. Sanity (100k)
    ds_sanity = ds_shuffled.select(range(min(SANITY_SIZE, total_len)))
    save_to_text(ds_sanity, "sanity", "train")

    # B. Test (5k)
    test_start = max(0, total_len - TEST_SIZE)
    ds_test = ds_shuffled.select(range(test_start, total_len))
    save_to_text(ds_test, "test", "test")

    # C. Production (1.5M)
    remain_end_idx = min(SANITY_SIZE + MAIN_TRAIN_SIZE, test_start)
    if remain_end_idx > SANITY_SIZE:
        ds_prod = ds_shuffled.select(range(SANITY_SIZE, remain_end_idx))
        save_to_text(ds_prod, "bist_1.5m", "train")
    else:
        print("Warning: Not enough data for production set.")

    print("All Done! Data is ready in'./data/' folder.")


if __name__ == "__main__":
    main()
