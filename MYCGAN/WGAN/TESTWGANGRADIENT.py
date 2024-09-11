"""
This script tests a pre-trained Conditional Generative Adversarial Network (cGAN) on the CelebA dataset.

### Key Components:
1. **Parameters**:
   - `dataroot`: Path to the CelebA image directory.
   - `attr_path`: Path to the CelebA attribute file.
   - `selected_attrs`: List of attributes used for conditional generation.
   - `path_model`: Path to the pre-trained generator model.
   - `save_img`: Directory where output images will be saved.
   - Other parameters include `batch_size`, `image_size`, `nz`, `ngf`, `ndf`, and `label_dim`.

2. **Data Loader**:
   - Uses `get_loader` function to create a data loader for the CelebA dataset in test mode.

3. **Image Normalization**:
   - `denorm(x)`: Converts images from the range [-1, 1] to [0, 1].

4. **Model Initialization**:
   - Defines and loads a pre-trained `Generator` model.

5. **Label Creation**:
   - `create_labels(c_org, c_dim, selected_attrs)`: Generates target domain labels for testing.

6. **Image Translation**:
   - For each batch of images:
     - Generates random noise and labels for conditional generation.
     - Translates images using the generator and adjusts dimensions.
     - Saves the generated images.

7. **Saving Results**:
   - Concatenates and saves real and fake images as a single output image.

### Usage:
1. Load a pre-trained generator model.
2. Run the script to generate and save images with different attributes applied.
"""

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
import torch.nn.functional as F  # Add this import
from dataloaderWGAN import get_loader
from generatorWGAN import Generator
from itertools import islice
from torch.utils.tensorboard import SummaryWriter


# Parameters
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/images'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/list_attr_celeba.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Attractive','Big_Lips', 
                  'Black_Hair','Eyeglasses', 'Blond_Hair','5_o_Clock_Shadow','Arched_Eyebrows','Attractive']  
path_model = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/generatorHPCWGAN2NEW2_25.pth'
save_img ='C:/Users/fstef/Desktop/PROGETTO_PRATI'

batch_size = 8
image_size = 64
nz = 128
ngf = 64
ndf = 64
label_dim = len(selected_attrs)

writer = SummaryWriter(log_dir='runs/WGANGRADIENT')

def test():

    
    mode = 'test'
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
        
        # This part of the code is performing image translation using a conditional Generative
        # Adversarial Network (GAN). 
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
            
            result_path = os.path.join(save_img, '{}-imagesWGANGRADIENT.jpg')
            vutils.save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))
            
            grid_images = vutils.make_grid(x_concat_denorm, nrow=8, normalize=True)  # Single row per batch
            writer.add_image('Real and Generated Images', grid_images )
            print(f'Saved and logged real and generated images into TensorBoard and {result_path}')
            
            writer.close()

