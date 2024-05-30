import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','model')))
from networks.DeepLabv3_plus import DeepLabV3Plus
from utils.utils import med_reshape

class DeepLabV3PlusInferenceAgent:
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):
        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = DeepLabV3Plus(num_classes=3, in_channels=1)
        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))
        self.model.to(device)
        self.model.eval()

    def single_volume_inference_unpadded(self, volume):
        raise NotImplementedError

    def single_volume_inference(self, volume):
        slices = np.zeros(volume.shape)

        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx, :, :]
            slice_2d = slice_2d.astype(np.single) / 255.0
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  

            prediction = self.model(slice_tensor.to(self.device))
            pred = np.squeeze(prediction.cpu().detach())

            slices[slice_idx, :, :] = torch.argmax(pred, dim=0).numpy()

        return slices
        
# if __name__ == "__main__":
#     # Example volume data (you should replace this with your actual volume data)
#     volume = np.random.randint(0, 256, (34, 49, 29), dtype=np.uint8)

#     # Replace 'path_to_saved_model.pth' with the actual path to your saved model file
#     model_path = 'path_to_saved_model.pth'

#     # Initialize the inference agent
#     inference_agent = DeepLabV3PlusInferenceAgent(
#         parameter_file_path=model_path,
#         model=DeepLabV3Plus(num_classes=3, in_channels=1),
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )

#     # Perform inference on the volume
#     segmentation = inference_agent.single_volume_inference(volume)
#     print(f'Segmentation shape: {segmentation.shape}')
