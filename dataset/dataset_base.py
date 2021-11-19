from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self):
        super(DatasetBase, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_data_loader(self, distributed=False):
        raise NotImplementedError
