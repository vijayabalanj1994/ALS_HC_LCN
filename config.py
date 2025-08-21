import torch

class Config():

    random_seed = 42
    dataset_dir_path = r"C:\Users\vijay\Neuro_BioMark\Hierarchical_Approaches\LCN\dataset\image_keys.xlsx"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {device} device")

config = Config()
