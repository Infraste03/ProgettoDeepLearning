"""
This class defines a Generator model for a Conditional Generative Adversarial Network (cGAN) in 
PyTorch. The generator is responsible for creating synthetic images based on random noise and 
conditional labels.

### Key Components:
1. **Initialization (`__init__`)**:
   - **Parameters**:
     - `nz`: Number of input noise dimensions.
     - `ngf`: Number of filters in the last convolutional layer.
     - `label_dim`: Number of dimensions for conditional labels.
   - **Network Architecture**:
     - **Noise Block**: Processes the random noise input. It consists of:
       - `nn.ConvTranspose2d`: Transpose convolution to upsample the noise.
       - `nn.BatchNorm2d`: Batch normalization for stabilizing training.
       - `nn.ReLU`: Activation function for introducing non-linearity.
     - **Label Block**: Processes the conditional labels. It is similar to the noise block and consists of:
       - `nn.ConvTranspose2d`: Transpose convolution to upsample the labels.
       - `nn.BatchNorm2d`: Batch normalization.
       - `nn.ReLU`: Activation function.
     - **Main Block**: Consists of multiple transpose convolution layers to progressively upsample 
       the concatenated noise and label feature maps to the final image size. The layers include:
       - `nn.ConvTranspose2d`: Transpose convolutions to generate the image.
       - `nn.BatchNorm2d`: Batch normalization.
       - `nn.ReLU`: Activation function.
       - `nn.Tanh`: Activation function for output layer to ensure pixel values are in range [-1, 1].

2. **Forward Pass (`forward`)**:
   - **Inputs**:
     - `noise`: Tensor of random noise with shape `[batch_size, nz, 1, 1]`.
     - `labels`: Tensor of conditional labels with shape `[batch_size, label_dim, 1, 1]`.
   - **Process**:
     - Passes `noise` through the `noise_block` to generate feature maps.
     - Passes `labels` through the `label_block` to generate label feature maps.
     - Resizes both outputs to `(4, 4)` to ensure they can be concatenated.
     - Concatenates the feature maps from `noise_block` and `label_block` along the channel dimension.
     - Passes the concatenated feature maps through the `main` block to produce the final image.
   - **Output**:
     - Generates a synthetic image with shape `[batch_size, 3, image_size, image_size]`, where `3` represents the RGB channels.

### Usage:
This `Generator` class is used in a conditional GAN setting to generate images based on noise and 
conditional labels. It takes noise and labels as input, processes them through separate blocks, 
concatenates the processed features, and outputs a synthetic image.
"""

# The above class defines a Generator model for a conditional Generative Adversarial Network (GAN) in
# PyTorch.
import torch
import torch.nn as nn
import torch.utils.data



# Define the Generator and Discriminator models
class Generator(nn.Module):
    def __init__(self,nz,ngf,label_dim):
        self.nz = nz
        self.ngf = ngf
        self.label_dim = label_dim
        
        super(Generator, self).__init__()
        
        self.noise_block = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
        
        self.label_block = nn.Sequential(
            nn.ConvTranspose2d(self.label_dim, self.ngf * 4, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        z_out = torch.nn.functional.interpolate(z_out, size=(4, 4))
        l_out = torch.nn.functional.interpolate(l_out, size=(4, 4))

        x = torch.cat([z_out, l_out], dim=1)
        return self.main(x)