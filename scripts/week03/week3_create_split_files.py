import pandas as pd
from pathlib import Path


def main():
    data_dir = Path(r"/data/processed/nist_sd300b")
    pos_path = data_dir / "pairs_pos.csv"
    neg_path = data_dir / "pairs_neg.csv"

    print("Loading datasets...")
    df_pos = pd.read_csv(pos_path)
    df_neg = pd.read_csv(neg_path)

    for split_name in ['train', 'val', 'test']:
        print(f"Processing split: {split_name}...")

        subset_pos = df_pos[df_pos['split'] == split_name]
        subset_neg = df_neg[df_neg['split'] == split_name]

        if len(subset_pos) == 0 and len(subset_neg) == 0:
            print(f"  Skipping {split_name} (no data found)")
            continue

        df_combined = pd.concat([subset_pos, subset_neg], ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

        out_file = data_dir / f"pairs_{split_name}.csv"
        df_combined.to_csv(out_file, index=False)

        print(f"  Saved: {out_file.name} (Total pairs: {len(df_combined)})")


if __name__ == "__main__":
    main()