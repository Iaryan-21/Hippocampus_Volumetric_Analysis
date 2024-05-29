import torch
from torch.utils.data import Dataset

class Slicesdataset(Dataset):
    def __init__(self, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data = data
        self.device = device
        self.slices = []
        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx
        # Ensure the image and seg tensors have a single channel
        sample['image'] = torch.from_numpy(self.data[slc[0]]['image'][slc[1], :, :]).unsqueeze(0).to(self.device, dtype=torch.float)
        sample['seg'] = torch.from_numpy(self.data[slc[0]]['seg'][slc[1], :, :]).unsqueeze(0).to(self.device, dtype=torch.long)

        return sample

    def __len__(self):
        return len(self.slices)
