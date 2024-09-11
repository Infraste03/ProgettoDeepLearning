"""
This script is designed for testing a pre-trained Generative Adversarial Network (GAN) model on the CelebA dataset. The script performs the following tasks:

1. **Setup and Imports**:
   - Imports necessary Python libraries and modules for data processing, neural network operations, and image handling.

2. **Parameters**:
   - Defines paths to the dataset, attribute files, and the pre-trained model.
   - Sets various hyperparameters and configurations including image size, batch size, and attribute dimensions.

3. **Function `test()`**:
   - **Data Loading**: Loads the CelebA dataset with specific attributes using a data loader.
   - **Model Initialization**: Creates a generator model and loads pre-trained weights.
   - **Label Generation**: Creates target labels for different attributes to test various conditions.
   - **Inference**: Generates new images using the pre-trained model, applying different attributes to the generated images.
   - **Saving Results**: Saves the generated images along with real images to a specified file path.

The script aims to demonstrate how the model translates images with specific attributes and saves the results for visual inspection.
"""

# The `import` statements at the beginning of the code are used to import various Python libraries and
# modules that are required for the functionality of the script.
# import statement:
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torchvision.transforms as transforms
import torch.nn.functional as F  
from generator import Generator
from dataloader import get_loader
from itertools import islice

from torch.utils.tensorboard import SummaryWriter


# Parameters
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/images'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/list_attr_celeba.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Wearing_Hat', 'Blond_Hair', 'Goatee', 'Chubby']  
path_model = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/generatorCOMBINATIO_4.pth'
save_img ='C:/Users/fstef/Desktop/PROGETTO_PRATI'

batch_size = 8
image_size = 64
nz = 100
ngf = 64
ndf = 64
label_dim = len(selected_attrs)

writer = SummaryWriter(log_dir='runs/COMBINATIONTEST')


def test():
    """
    The function `test()` performs image translation using a pre-trained model on the CelebA dataset
    with specific attributes and settings.
    """

    
    mode = 'test'
    # The line `celeba_loader = get_loader(dataroot, attr_path, selected_attrs, 178, image_size,
    # batch_size, 'CelebA', mode, 2)` is creating a data loader object for loading and iterating over
    # the CelebA dataset with specific attributes and settings. 
    celeba_loader = get_loader(dataroot, attr_path, selected_attrs,
                            178, image_size, batch_size,
                            'CelebA', mode, 2)


    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    to_pil = transforms.ToPILImage()

    fixed_batch, fixed_c = next(iter(celeba_loader))

    print(selected_attrs)


    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    G = Generator(nz,ngf,label_dim).to(device)

    # load the pretrained weights
    
    G.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))#*


    def create_labels(c_org, c_dim, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Smiling', 'Young', 'Male', 'Attractive','Big_Lips', 
                  'Black_Hair','Eyeglasses', 'Blond_Hair','5_o_Clock_Shadow','Arched_Eyebrows','Attractive']  :
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(device))
        return c_trg_list


    # now we can do the inference 

    # Inference code with resizing to ensure consistency in dimensions
    with torch.no_grad():
        # The line `for i, (x_real, c_org) in enumerate(celeba_loader):` is iterating over the
        # `celeba_loader` which is a data loader object that provides batches of data during training
        # or testing.
        # The line `for i, (x_real, c_org) in enumerate(celeba_loader):` is iterating over the
        # `celeba_loader` which is a data loader object that provides batches of data during training
        # or testing.
        for (x_real, c_org) in islice(celeba_loader, 1):
            # Generate random noise and labels
            noise = torch.randn(x_real.size(0), nz, 1, 1, device=device)
            c_trg_list = create_labels(c_org, label_dim, selected_attrs)
            
            # Translate images.
            x_fake_list = [x_real.to(device)]
            for c_trg in c_trg_list:
                c_trg = c_trg.unsqueeze(2).unsqueeze(3)  # Adjust shape for broadcasting
                noise_input = noise  # Use the noise as input
                x_fake = G(noise_input, c_trg)
                
                # Resize to 128x128
                x_fake_resized = F.interpolate(x_fake, size=(image_size, image_size))
                x_fake_list.append(x_fake_resized)

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            x_concat_denorm = denorm(x_concat)
            result_path = os.path.join(save_img, '{}-imagesCOMBINATIONLOSS.jpg')
            vutils.save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))
            
            grid_images = vutils.make_grid(x_concat_denorm, nrow=8, normalize=True)  # Single row per batch
            writer.add_image('Real and Generated Images', grid_images )
            print(f'Saved and logged real and generated images into TensorBoard and {result_path}')
            
            writer.close()

