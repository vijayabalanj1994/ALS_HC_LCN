from pathlib import Path
from config import config

def get_img_paths_and_labels(train_df, test_df, target_label):
    IMG_COL = "Image No"
    LABEL_COL = "Category"

    img_dir = Path(config.img_dir_path)

    # image paths
    train_img_path  = [str(img_dir / f"{int(x)}.tif") for x in train_df[IMG_COL]]
    test_image_path = [str(img_dir / f"{int(x)}.tif") for x in test_df[IMG_COL]]

    # binary labels: target -> 1, others -> 0
    train_lables = [1 if c == target_label else 0 for c in train_df[LABEL_COL]]
    test_lable  = [1 if c == target_label else 0 for c in test_df[LABEL_COL]]

    return train_img_path, test_image_path, train_lables, test_lable

if __name__ == "__main__":

    # getting the train and test df of fold 0
    from src.prepare_data.prepare_folds import get_folds
    folds = get_folds()
    train_df, test_df = folds[0]["train_df"], folds[0]["test_df"]

    train_img_path, test_image_path, train_lables, test_lable = get_img_paths_and_labels(train_df, test_df, target_label="Control")
    print(test_image_path)
    print(test_lable)

