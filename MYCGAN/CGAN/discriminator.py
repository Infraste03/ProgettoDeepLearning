"""
The `Discriminator` class defines a neural network model used in the Generative Adversarial Network (GAN) framework. 

**Purpose:**
- The Discriminator's main function is to differentiate between real and fake images. It evaluates whether an input image is a real image from the dataset or a fake image generated by the Generator. Additionally, this Discriminator is designed to incorporate conditional information through labels, allowing it to assess whether the generated image matches specific attributes defined by the labels.

**Architecture:**
- The `Discriminator` is composed of the following main components:

  1. **Image Block (`img_block`):**
     - Takes an input image and applies a convolutional layer followed by a LeakyReLU activation function. This block helps in extracting features from the image.

  2. **Label Block (`label_block`):**
     - Processes the conditional labels using a convolutional layer and a LeakyReLU activation function. This block helps in integrating label information with the image features.

  3. **Main Block (`main`):**
     - This sequential block consists of several convolutional layers interspersed with Batch Normalization and LeakyReLU activations. It refines the features and produces a final output. The final layer uses a Sigmoid activation function to output a probability value indicating whether the input is real or fake.

**Forward Pass:**
- In the `forward` method:
  - The input image and label are separately processed by their respective blocks (`img_block` and `label_block`).
  - The outputs of these blocks are concatenated along the channel dimension.
  - The concatenated tensor is then passed through the `main` block.
  - The final output is reshaped and averaged to produce a single probability value for each input, representing the Discriminator's assessment of the input image.

**Key Features:**
- **Conditional Discrimination:** The Discriminator uses label information to condition its evaluation, enabling it to assess whether the image meets specific conditions based on the labels.
"""

# The `Discriminator` class in the provided code is a neural network model for image classification
# with conditional information using convolutional layers and leaky ReLU activations.
#The discriminator is responsible for evaluating the images generated by the generator, establishing whether they are real or fake. 
# The special feature of this discriminator is that it is conditioned by the labels, which means that it checks whether the image fulfils certain conditions (provided by the labels).
import torch
import torch.nn as nn
import torch.utils.data

class Discriminator(nn.Module):
        def __init__(self, ndf, label_dim):
            
            self.ndf = ndf
            self.label_dim = label_dim
            super(Discriminator, self).__init__()

            self.img_block = nn.Sequential(
                nn.Conv2d(3, self.ndf // 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.label_block = nn.Sequential(
                nn.Conv2d(self.label_dim, self.ndf // 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.main = nn.Sequential(
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, img, label):
            img_out = self.img_block(img)
            lab_out = self.label_block(label)
            x = torch.cat([img_out, lab_out], dim=1)
            x = self.main(x)
            return x.view(img.size(0), -1).mean(dim=1)
