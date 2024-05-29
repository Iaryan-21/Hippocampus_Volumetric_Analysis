import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.slices import Slicesdataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d
from networks.RecursiveUNet import RecursiveUNet
from networks.UNet import IUNet 
from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:

    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        self.train_loader = DataLoader(Slicesdataset(dataset[split["train"]]),batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(Slicesdataset(dataset[split["val"]]),batch_size=config.batch_size, shuffle=True, num_workers=0)

        self.test_data = dataset[split["test"]]

        if not torch.cuda.is_available():
            print("Lets go CPU!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.model = RecursiveUNet(in_channels=1, num_classes=3)
        self.model = IUNet(in_channels=1, num_classes=3)
        self.model.to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def save_images(self, epoch, data, target, prediction, batch_idx, stage='train'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(data[0, 0].cpu(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(target[0, 0].cpu(), cmap='gray')  # Changed target shape
        axes[1].set_title('Label')
        axes[2].imshow(torch.argmax(prediction[0], dim=0).cpu(), cmap='gray')
        axes[2].set_title('Prediction')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.out_dir, f"{stage}_epoch_{epoch}_batch_{batch_idx}.png")
        plt.savefig(save_path)
        plt.close(fig)

    def train(self):
        print(f"Training Epoch {self.epoch}")
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = batch['image'].to(self.device, dtype=torch.float)
            target = batch['seg'].to(self.device, dtype=torch.long)

            print(f"Batch {i} - Data shape: {data.shape}, Target shape: {target.shape}")

            prediction = self.model(data)

            loss = self.loss_function(prediction, target[:, 0, :, :])

            print(f"Batch {i} - Prediction shape: {prediction.shape}, Loss: {loss.item()}")

            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                print(f"\nEpoch : {self.epoch}, Train Loss: {loss.item()}, {100 * (i + 1) / len(self.train_loader):.1f}% complete")
                counter = 100 * self.epoch + 100 * (i / len(self.train_loader))
                log_to_tensorboard(self.tensorboard_train_writer, loss, data, target, prediction, counter)
                self.save_images(self.epoch, data, target, prediction, i, stage='train')

            print(".", end='')
        print("\nTraining Complete")

    def validate(self):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                data = batch['image'].to(self.device, dtype=torch.float)
                target = batch['seg'].to(self.device, dtype=torch.long)

                print(f"Validation Batch {i} - Data shape: {data.shape}, Target shape: {target.shape}")

                prediction = self.model(data)
                loss = self.loss_function(prediction, target[:, 0, :, :])

                print(f"Validation Batch {i} - Prediction shape: {prediction.shape}, Loss: {loss.item()}")

                loss_list.append(loss.item())

                if (i % 10) == 0:
                    self.save_images(self.epoch, data, target, prediction, i, stage='val')

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction,
            (self.epoch + 1) * 100
        )
        print("Validation is Complete")

    def save_model_parameters(self):
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(),path)

    def load_model_parameters(self, path=''):
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find the model path {model_path}")
    
    def run_test(self):
        print("Testing:")
        self.model.eval()
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)
        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])
            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc
                })
            
            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

            out_dict["overall"] = {
                "mean_dice": np.mean(dc_list),
                "mean_jaccard": np.mean(jc_list)}

            print("\nTesting complete.")
            return out_dict

    def run(self):
        self._time_start = time.time()
        print("Training for UNET has commenced:")
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        self.save_model_parameters()
        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
