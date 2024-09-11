"""
The `Discriminator` class defines a neural network used in Generative Adversarial Networks (GANs) to distinguish between real and generated (fake) images. It processes both image and label inputs to make its classification.

**Architecture:**

1. **Image Processing Block (`img_block`):**
   - This block handles the image input.
   - It consists of three convolutional layers, each followed by a LeakyReLU activation. Batch Normalization is applied after the second and third convolutional layers to stabilize training and improve convergence.
   - The layers progressively extract and refine features from the input image.

2. **Label Processing Block (`label_block`):**
   - This block processes the label input.
   - It is structured similarly to the image block, with three convolutional layers followed by LeakyReLU activations and Batch Normalization.
   - The label input is expected to have multiple channels (e.g., 5), which are reduced to the same feature dimensions as the image block.

3. **Feature Integration (`main`):**
   - After processing the image and label inputs, their outputs are concatenated along the channel dimension.
   - This concatenated tensor is then passed through additional convolutional layers that reduce the combined feature dimensions and further refine the features.

4. **Classification (`classifier`):**
   - The final classification is performed through a convolutional layer followed by a Sigmoid activation.
   - This outputs a single channel indicating the probability that the input is real or fake.

**Forward Pass:**

- The `forward` method processes the input image and label through their respective blocks to obtain feature maps.
- If necessary, the label feature maps are resized to match the dimensions of the image feature maps.
- The feature maps from both the image and label blocks are concatenated and passed through the `main` block to extract the final features.
- The network returns two outputs:
  1. `real_fake_output`: A scalar value representing the network's estimate of whether the input is real or fake.
  2. `class_output`: A tensor with the same spatial dimensions as the input, used to classify the input as real or fake.

**Key Features:**

- **Conditional Processing:** The Discriminator uses label information to condition its evaluation, helping it to assess whether the image meets certain conditions specified by the labels.
- **Feature Concatenation:** Image and label features are combined to provide a more comprehensive evaluation of the input data.
"""

# The `Discriminator` class in this code defines a neural network model for image and label processing
# with a final output for real/fake classification.
import torch
import torch.nn as nn
import torch.utils.data



class Discriminator(nn.Module):
        def __init__(self, ndf, label_dim):
            self.ndf = ndf
            self.label_dim = label_dim
            super(Discriminator, self).__init__()
            
            # Image processing block
            self.img_block = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),  # Expecting 3 channels for the image (RGB)
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Label processing block (adjusted to expect 5 channels for the label input)
            self.label_block = nn.Sequential(
                nn.Conv2d(self.label_dim, 64, 4, 2, 1),  
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )


            self.main = nn.Sequential(
                nn.Conv2d(512, 256, 4, 2, 1),  # Concatenate img and label blocks, hence 512 input channels
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.classifier = nn.Sequential(
                nn.Conv2d(256, 1, 4),  # Output 1 channel for real/fake classification
                nn.Sigmoid()
            )

        def forward(self, img, label):
            img_out = self.img_block(img)
            lab_out = self.label_block(label)

            # Ensure both img_out and lab_out are the same size before concatenation
            if img_out.size(2) != lab_out.size(2) or img_out.size(3) != lab_out.size(3):
                lab_out = torch.nn.functional.interpolate(lab_out, size=img_out.size()[2:])

            x = torch.cat([img_out, lab_out], dim=1)
            features = self.main(x)

            real_fake_output = features.view(img.size(0), -1).mean(dim=1)
            class_output = self.classifier(features)

            return real_fake_output, class_output