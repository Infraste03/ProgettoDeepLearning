"""
This script implements a conditional GAN (Generative Adversarial Network) model for image generation, using a dataset of faces (CelebA). The model is conditioned on selected attributes (such as 'Smiling', 'Male', etc.) to generate images that reflect those attributes.

The code is divided into two main operations: 'train' and 'test', which can be specified via command-line arguments.
- In 'train' mode, the script:
  1. Loads the dataset with images and attribute labels using a custom DataLoader.
  2. Defines the Generator network and the Discriminator network.
  3. Uses a binary loss function (BCELoss) to compare real and generated images, and a classification loss (BCEWithLogitsLoss) to predict attributes of real images.
  4. Updates the Discriminator using both real and generated (fake) images, calculating losses for each case.
  5. Updates the Generator to "fool" the Discriminator by generating increasingly realistic images.
  6. Displays some example images and their predicted labels, and periodically saves the trained models.

- In 'test' mode, the script:
  - Runs the model test by calling the 'test' function (implemented in the TESTCOMBINATION file).

Parameters such as batch size, number of epochs, dataset path, and model architectures are defined in the initial section of the code.
"""

# The `import` statements at the beginning of the code are used to import necessary libraries and
# modules in Python for various functionalities:
import argparse
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
from TESTCOMBINATION import test
from dataloader import get_loader
from generator import Generator
from combination_discriminator import Discriminator
#dos2unix sbatch.sh

# Parameters
# The variables defined at the beginning of the code are setting up various parameters and paths
# required for training a Generative Adversarial Network (GAN) model for image generation.
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/DFSmall/minnie'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/attributeSmall.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Wearing_Hat', 'Blond_Hair', 'Goatee', 'Chubby']  
batch_size = 128
image_size = 64
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1
label_dim = len(selected_attrs)


if __name__ == '__main__':
    # Create the dataset and dataloader using the new method
    
    # The above code is a Python script using the `argparse` module to create a command-line interface
    # for a script that trains and tests a model. It defines a parser with a description for the
    # script, adds an argument 'operation' with choices 'train' or 'test' to specify whether to
    # perform training or testing, and then parses the command-line arguments provided when running
    # the script. If the 'operation' argument is 'train', it will execute the code block under that
    # condition.
    parser = argparse.ArgumentParser(description="TRAIN OR TEST?")
    parser.add_argument('operation', choices=['train', 'test'], help="TRAIN OR TEST?")
    args = parser.parse_args()

    if args.operation == 'train':
        # The above code snippet is creating a data loader object for training a model. It is using
        # the `get_loader` function with the specified parameters such as the root directory of the
        # data (`dataroot`), path to the attributes file (`attr_path`), selected attributes
        # (`selected_attrs`), image size, batch size, mode set to 'train', and number of workers for
        # data loading. This data loader will be used to load and preprocess the training data for the
        # model.
        dataloader = get_loader(dataroot, attr_path, selected_attrs, image_size=image_size, 
                                batch_size=batch_size, mode='train', num_workers=2)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        print("Using device:", device)

        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        #plt.show()

        # Create the Generator and Discriminator
        netG = Generator(nz, ngf, label_dim).to(device)
        netD = Discriminator(ndf,label_dim).to(device)

        # Apply the weights_init function to randomly initialize all weights
        
        def weights_init(m):
            """
            The function `weights_init` initializes weights for Convolutional and Batch Normalization
            layers in a neural network.
            
            :param m: The `weights_init` function is used to initialize the weights of a neural network
            model. It takes a module `m` as input, which represents a layer or a block of layers in the
            neural network
            """
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        netG.apply(weights_init)
        netD.apply(weights_init)

        # Create the loss function and optimizers
        
       
        # `criterion = nn.BCELoss()` is defining the binary cross-entropy loss function (`BCELoss`)
        # from the `torch.nn` module in PyTorch.
        criterion = nn.BCELoss()
        # `classification_criterion = nn.BCEWithLogitsLoss()` is defining a loss function for
        # classification tasks in PyTorch.
        classification_criterion = nn.BCEWithLogitsLoss()

        # `optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))` is creating an
        # Adam optimizer for the Discriminator neural network (`netD`).
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        # `optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))` is creating an
        # Adam optimizer for the Generator neural network (`netG`).
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Training Loop 
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
    # Update Discriminator
                netD.zero_grad()
                real, labels = data
                batch_size = real.size(0)
                real = real.to(device)
                labels = torch.cat([labels] * (image_size // 64), dim=0).unsqueeze(2).unsqueeze(3).expand(-1, -1, image_size, image_size).to(device)
                real_labels = labels.to(device)

                # Real data adversarial loss
                label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                real_output, class_output_real = netD(real, real_labels)
                errD_real = criterion(real_output, label)

                # Real data classification loss
                real_labels_reduced = labels.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)  
                if real_labels_reduced.size(1) != 1:
                    real_labels_reduced = real_labels_reduced.mean(dim=1, keepdim=True)  
                class_loss_real = classification_criterion(class_output_real, real_labels_reduced)

                # Total discriminator loss for real images
                total_errD_real = errD_real + class_loss_real
                total_errD_real.backward()
                D_x = real_output.mean().item()

                # Print average of predicted labels from discriminator
                average_pred_labels = class_output_real.view(-1).mean().item()
                #print(f'Average predicted labels (real images): {average_pred_labels}')

                # Fake data adversarial loss
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise, labels)
                label.fill_(0)
                fake_output, _ = netD(fake.detach(), labels)
                errD_fake = criterion(fake_output, label)
                errD_fake.backward()
                D_G_z1 = fake_output.mean().item()
                optimizerD.step()

                # Update Generator
                netG.zero_grad()
                label.fill_(1)
                fake_output, _ = netD(fake, labels)
                errG = criterion(fake_output, label)
                errG.backward()
                D_G_z2 = fake_output.mean().item()
                optimizerG.step()

                print(f'[{epoch+1}/{num_epochs}] '
                    f'[{i}/{len(dataloader)}] '
                    f'D_loss_real: {errD_real.item()} '
                    f'D_loss_fake: {errD_fake.item()} '
                    f'class_loss_real: {class_loss_real.item()} '
                    f'G_loss: {errG.item()} '
                    f'D(x): {D_x} '
                    f'D(G(z)): {D_G_z1} / {D_G_z2}')



            
            if(epoch % 2 == 0):
            # Save the models
                torch.save(netG.state_dict(), f'generatorCOMBINATIOnew_{epoch+1}.pth')
                torch.save(netD.state_dict(), f'discriminatorCOMBINATIOnew_{epoch+1}.pth')

                # Save some samples
                
                # The code snippet  is responsible for generating and displaying sample
                # images along with their corresponding predicted labels during the training process
                # of a Generative Adversarial Network (GAN) model.
                with torch.no_grad():
                    sample_images = real[:16].cpu()  
                    sample_labels = class_output_real[:16].cpu()  

                    
                    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                    for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
                        ax = axes[i // 4, i % 4]
                        ax.imshow(np.transpose(image, (1, 2, 0)))
                        ax.set_title(f"Pred: {label.mean().item():.2f}")
                        ax.axis('off')
                    #plt.show()
    # The code snippet is checking if the value of `args.operation` is equal to the string 'test'. If
    # it is, then it calls the function `test()`.              
    elif args.operation == 'test':
        test()

                
            
