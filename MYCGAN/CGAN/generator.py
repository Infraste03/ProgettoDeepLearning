"""
The `Generator` class is a neural network module in PyTorch designed to generate images from random noise and label inputs. This class utilizes transposed convolutional layers to upsample the input features and create realistic images based on specified conditions. Here's a breakdown of its components:

1. **Initialization (`__init__` method)**:
   - **Parameters**:
     - `nz`: Dimension of the input noise vector.
     - `ngf`: Number of feature maps in the generator.
     - `label_dim`: Dimension of the label vector representing the characteristics of the generated image.
   - **Noise Block**:
     - A series of layers that transform the noise vector into feature maps. It includes a `ConvTranspose2d` layer followed by batch normalization and a ReLU activation function.
   - **Label Block**:
     - A series of layers that process the label vector to produce feature maps. This block also uses `ConvTranspose2d`, batch normalization, and ReLU activation.
   - **Main Network**:
     - Combines the feature maps from the noise and label blocks and transforms them into the final image. This sequence of transposed convolutions gradually upsamples the feature maps, with batch normalization and ReLU activation in between. The final output layer uses `Tanh` activation to produce images with pixel values in the range [-1, 1].

2. **Forward Pass (`forward` method)**:
   - **Inputs**:
     - `noise`: A tensor containing random noise used as input to the generator.
     - `labels`: A tensor containing labels that specify desired characteristics of the generated image.
   - **Process**:
     - The noise and labels are processed separately through their respective blocks.
     - The outputs from the noise and label blocks are upsampled to match their dimensions.
     - The processed noise and label features are concatenated along the channel dimension.
     - The concatenated features are passed through the main network to produce the final generated image.
"""

# The class `Generator` is a neural network module in PyTorch that generates images based on input
# noise and labels through a series of transposed convolutional layers.
#The generator takes noise and labels as input and generates realistic images that comply with the conditions of the labels.
# The noise introduces randomness into the process, while the labels specify the desired characteristics of the image (e.g. a smiling or spectacled face).
import torch
import torch.nn as nn
import torch.utils.data



class Generator(nn.Module):
        def __init__(self, nz, ngf, label_dim):
            self.nz = nz # dimension noise input 
            self.ngf = ngf #number feature maps 
            self.label_dim = label_dim # number of label 
            
            super(Generator, self).__init__()
            
            self.noise_block = nn.Sequential(
                #trasfroming random noise into output of feature maps 
                nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0, bias=False),  
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True)
            )
            
            self.label_block = nn.Sequential(
                #acts on the labels 
                nn.ConvTranspose2d(self.label_dim, self.ngf * 4, 4, 1, 0, bias=False), 
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True)
            )
            
            self.main = nn.Sequential(
                #form feature maps combined trasfroming into image
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
            z_out = self.noise_block(noise) #elaboration of the noise 
            l_out = self.label_block(labels) #elaboration of the labels
            
            z_out = torch.nn.functional.interpolate(z_out, size=(4, 4))
            l_out = torch.nn.functional.interpolate(l_out, size=(4, 4))

            x = torch.cat([z_out, l_out], dim=1)
            return self.main(x)