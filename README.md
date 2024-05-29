# HIPPOCAMPAL VOLUMETRIC QUANTIFICATION OF ALZHEIMER'S PROGRESSION

## Segmentation using the UNET, DEEPLABv3+ and RecursiveUNet architecture

### Aim
This algorithm is intended to help diagnose and track any progression of Alzheimer's disease and forms of dementia in patients displaying symptoms of disease including short and long term memory loss issues as well. This quantification will help in disease management.

### System Design

UNET is an end-to-end AI system which features a Deep Neural Network algorithm that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients as their studies are committed to the clinical imaging archive.

### UNET Architecture:
![UNET Architecture](https://github.com/Iaryan-21/Hippocampus_Volumetric_Analysis/blob/main/unet_arch.png)

### Architecture UNET
UNET is a type of Convolutional Neural Network (CNN) primarily used for image segmentation. It was first introduced in 2015 for biomedical image segmentation. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

1. **Contracting Path (Encoder)**:
   - The left side of the U-shaped architecture.
   - Consists of repeated application of two 3x3 convolutions (unpadded), each followed by a ReLU and a 2x2 max pooling operation for downsampling.
   - At each downsampling step, the number of feature channels is doubled.

2. **Bottleneck**:
   - The bottom of the U-shape, where the resolution of the feature maps is the lowest, but the highest number of feature channels are present.

3. **Expanding Path (Decoder)**:
   - The right side of the U-shaped architecture.
   - Each step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels.
   - Then, a concatenation with the correspondingly cropped feature map from the contracting path.
   - Followed by two 3x3 convolutions, each followed by a ReLU.

4. **Output Layer**:
   - A final 1x1 convolution is used to map each 64-component feature vector to the desired number of classes.

The UNET architecture allows for precise localization and efficient feature extraction, making it highly effective for segmentation tasks, particularly in medical imaging where it can delineate structures such as the hippocampus for volumetric analysis.

### Dataset

The hippocampus training data was gather from the Medical Decathlon competition, found at
http://www.medicaldecathlon.com. This dataset is stored as a collection of 260 Neuroimaging
Informatics Technology Initiative (NIfTI) files, with one file per image volume, and one file per corresponding segmentation mask. The original images are T2 MRI scans of the full brain. This dataset utilizes cropped volumes where the region around the hippocampus has been cut out.

### Output

![Output Sample](https://github.com/Iaryan-21/Hippocampus_Volumetric_Analysis/blob/main/val_epoch_77_batch_0.png)

### Results

Here are the key results from the analysis:

- **Filename**: hippocampus_328.nii
- **Dice Coefficient**: 0.8700673724735323
- **Jaccard Index**: 0.7700170357751278

Overall statistics:

- **Mean Dice Coefficient**: 0.8700673724735323
- **Mean Jaccard Index**: 0.7700170357751278

Configuration:

- **Model Name**: UNet
- **Root Directory**: `C:\\Users\\aryan\\OneDrive\\Desktop\\Hypocampal Volume Quantification of Alzheimers\\model\\data\\`
- **Number of Epochs**: 100
- **Learning Rate**: 0.0001
- **Batch Size**: 8
- **Patch Size**: 64
- **Test Results Directory**: `./test_results`
