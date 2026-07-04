import pandas as pd
from sklearn.model_selection import train_test_split

def create_lesion_splits(metadata_df, random_state=10, val_size=0.15, test_size=0.15, tolerance=0.03):
    lesion_df = metadata_df.drop_duplicates(subset='lesion_id')[['lesion_id', 'dx']]
    train_val_ids, test_ids = train_test_split(
        lesion_df['lesion_id'], 
        test_size=test_size, 
        stratify=lesion_df['dx'], 
        random_state=random_state
    )
    train_val_df = lesion_df[lesion_df['lesion_id'].isin(train_val_ids)]
    train_ids, val_ids = train_test_split(
        train_val_df['lesion_id'],
        test_size= val_size / (1 - test_size),
        stratify=train_val_df['dx'],
        random_state=random_state
    )

    train_df = metadata_df[metadata_df['lesion_id'].isin(train_ids)]
    test_df = metadata_df[metadata_df['lesion_id'].isin(test_ids)]
    val_df = metadata_df[metadata_df['lesion_id'].isin(val_ids)]

    # --- Assertion 1: no lesion_id leaks across splits ---

    train_set = set(train_ids)
    test_set = set(test_ids)
    val_set = set(val_ids)

    assert train_set.isdisjoint(val_set), "Leakage: lesion_id(s) shared between train and val"
    assert train_set.isdisjoint(test_set), "Leakage: lesion_id(s) shared between train and test"
    assert val_set.isdisjoint(test_set), "Leakage: lesion_id(s) shared between val and test"

    print(f"✓ No lesion_id overlap across train/val/test")
    print(f"  Train: {len(train_set)} lesions, Val: {len(val_set)} lesions, Test: {len(test_set)} lesions")

    # --- Assertion 2: dx proportions roughly preserved in each split ---

    full_props = lesion_df['dx'].value_counts(normalize=True).sort_index()

    train_props = lesion_df[lesion_df['lesion_id'].isin(train_ids)]['dx'].value_counts(normalize=True).sort_index()
    val_props = lesion_df[lesion_df['lesion_id'].isin(val_ids)]['dx'].value_counts(normalize=True).sort_index()
    test_props = lesion_df[lesion_df['lesion_id'].isin(test_ids)]['dx'].value_counts(normalize=True).sort_index()

    for split_name, split_props in [('train', train_props), ('val', val_props), ('test', test_props)]:
        diff = (split_props - full_props).abs()
        max_diff = diff.max()
        assert max_diff < tolerance, (
            f"Class proportion mismatch in {split_name} split: "
            f"max deviation {max_diff:.3f} for class {diff.idxmax()}"
        )
        print(f"✓ {split_name} split class proportions within {tolerance:.0%} of full dataset "
            f"(max deviation: {max_diff:.3f})")
        
    return train_df, test_df, val_df