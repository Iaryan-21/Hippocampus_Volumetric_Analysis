import torch
import numpy as np
from networks.RecursiveUNet import RecursiveUNet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','model')))
from utils.utils import med_reshape

class UNetInferenceAgent:
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):
        self.model =  model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = RecursiveUNet(num_classes=3)
        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))
        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        raise NotImplementedError
    
    def single_volume_inference(self, volume):
        slices = np.zeros(volume.shape)
        
        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx, :, :]
            slice_2d = slice_2d.astype(np.single) / 255.0
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
            print('slice_tensor: ', slice_tensor.shape)
            
            prediction = self.model(slice_tensor.to(self.device)) 
            pred = np.squeeze(prediction.cpu().detach())
            print(f'Inference, prediction: {prediction.shape}, pred: {pred.shape}')
            
            slices[slice_idx, :, :] = torch.argmax(pred, dim=0).numpy() 
            
        return slices
