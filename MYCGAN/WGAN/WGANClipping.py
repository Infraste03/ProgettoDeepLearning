"""
This script implements a training pipeline for a Wasserstein Generative Adversarial Network (WGAN) using PyTorch. 

The script begins by importing necessary libraries and modules for deep learning and data handling. The core components of the WGAN are defined, including a Generator and Discriminator network. 

Parameters for training such as dataset paths, hyperparameters, and training settings are specified. The script then initializes TensorBoard for monitoring training progress and visualizing results.

The script supports two operations: 'train' and 'test'. 
- For training (`args.operation == 'train'`):
  1. **Data Loading**: It loads the dataset using a custom data loader function `get_loader`.
  2. **Device Selection**: It chooses whether to run the training on a GPU or CPU based on availability.
  3. **Model Initialization**: Initializes the Generator and Discriminator networks and applies weight initialization.
  4. **Optimizer Setup**: Sets up the optimizers for both the Generator and Discriminator.
  5. **Training Loop**: Runs the training process, where the Discriminator and Generator are updated iteratively. The Discriminator is updated multiple times per Generator update, and weight clipping is applied to maintain the Wasserstein loss constraints. Training statistics and sample images are periodically saved and logged to TensorBoard.
  6. **Model Saving**: Saves the trained models and generated images at regular intervals.

- For testing (`args.operation == 'test'`):
  - It simply calls a test function, which should contain code for evaluating the trained models.

This setup allows for comprehensive monitoring and evaluation of the WGAN's performance through TensorBoard and saved model checkpoints.
"""

# The code snippet you provided is importing necessary libraries and modules for a deep learning
# project using PyTorch framework. 
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
from dataloaderWGAN import get_loader
from generatorWGAN import Generator
from discriminatorWGAN import Discriminator
import argparse
from TESTWGANCLIPPING import test 

from torch.utils.tensorboard import SummaryWriter
from matplotlib.lines import Line2D

# Parameters
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/images'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/list_attr_celeba.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Attractive','Big_Lips', 
                  'Black_Hair','Eyeglasses', 'Blond_Hair','5_o_Clock_Shadow','Arched_Eyebrows','Attractive']
batch_size = 128 #256
image_size = 64
nz = 100 #256
ngf = 64
ndf = 64
num_epochs = 51
lr = 0.0002 #0.0002
beta1 = 0.5
ngpu = 1
label_dim = len(selected_attrs)
critic_iters = 5#10  # Number of Critic iterations per Generator iteration
clamp_lower = -0.01
clamp_upper = 0.01

writer = SummaryWriter('./runs/tensorWGAN1')


if __name__ == '__main__':
    # Create the dataset and dataloader using the new method
    
    parser = argparse.ArgumentParser(description="TRAIN OR TEST ?")
    parser.add_argument('operation', choices=['train', 'test'], help="TRAIN OR TEST ?")
    args = parser.parse_args()

    if args.operation == 'train':
        # The line `dataloader = get_loader(dataroot, attr_path, selected_attrs,
        # image_size=image_size, batch_size=batch_size, mode='train', num_workers=2)` is creating a
        # data loader for training data.
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
        
        ##########################
        img_grid = vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu()
        img_grid = img_grid.numpy()
        img_grid = np.transpose(img_grid, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
        # Add the image to TensorBoard
        writer.add_image('Example Star', img_grid, dataformats='HWC')
        
        def add_gradient_hist(net):
            ave_grads = [] 
            layers = []
            for n,p in net.named_parameters():
                if ("bias" not in n):
                    layers.append(n)
                    if p.requires_grad: 
                        ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
                    else:
                        ave_grad = 0
                    ave_grads.append(ave_grad)
                
            layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
            
            fig = plt.figure(figsize=(12, 12))
            plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
            plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
            plt.xlim(left=0, right=len(ave_grads))
            plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            #plt.grid(True)
            plt.legend([Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
            plt.tight_layout()
            #plt.show()
            
            return fig
        
        ###################################################

        # Create the Generator and Discriminator
        netG = Generator(nz,ngf,label_dim).to(device)
        netD = Discriminator(ndf,label_dim).to(device)

        # Apply the weights_init function to randomly initialize all weights
        def weights_init(m):
            """
            The function `weights_init` initializes weights for convolutional and batch normalization
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
                
        """ 
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                # Initialize Conv layers with Xavier initialization
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                # Initialize Linear layers with Xavier initialization, if present
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                # Usa Kaiming per layer di convoluzione con ReLU
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif classname.find('Linear') != -1:
                # Usa Xavier per layer lineari
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif classname.find('BatchNorm') != -1:
                # Inizializza BatchNorm: setta i pesi a 1 e i bias a 0
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif classname.find('LayerNorm') != -1:
                # Usa Xavier per LayerNorm
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))"""

        netG.apply(weights_init)
        netD.apply(weights_init)

        # Setup optimizer
        # The lines `optimizerD = optim.RMSprop(netD.parameters(), lr=lr)` and `optimizerG =
        # optim.RMSprop(netG.parameters(), lr=lr)` are setting up the optimizers for the Discriminator
        # (optimizerD) and Generator (optimizerG) networks, respectively, using the RMSprop optimizer
        # in PyTorch.
        optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
        optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

        # Training Loop
        print("Starting Training Loop...")
        # The above code is implementing a training loop for training a Generative Adversarial Network
        # (GAN) using the Wasserstein GAN (WGAN) algorithm. 
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                real, labels = data
                batch_size = real.size(0)
                real = real.to(device)
                labels = torch.cat([labels] * (image_size // 64), dim=0).unsqueeze(2).unsqueeze(3).expand(-1, -1, image_size, image_size).to(device)

                # Update Discriminator (Critic)
                for p in netD.parameters():
                    p.requires_grad = True

                for _ in range(critic_iters):
                    netD.zero_grad()
                    output_real = netD(real, labels)
                    errD_real = output_real.mean()

                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake = netG(noise, labels)
                    output_fake = netD(fake.detach(), labels)
                    errD_fake = output_fake.mean()
                    
                    D_loss = errD_fake - errD_real  # Loss del Discriminatore per WGAN
                    D_loss.backward()
                    optimizerD.step()
                    
                    # Clipping dei pesi del Discriminatore
                    for p in netD.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)

                D_x = errD_real.item()
                D_G_z1 = errD_fake.item()

                # Update Generator
                for p in netD.parameters():
                    p.requires_grad = False  # Evita la computazione
                netG.zero_grad()
                output = netD(fake, labels)
                G_loss = -output.mean()  # Loss del Generatore per WGAN
                G_loss.backward()
                optimizerG.step()
                
                writer.add_figure('gradients',
                                add_gradient_hist(netG),
                                global_step=epoch * len(dataloader) + i)
                writer.add_figure('gradients',
                                add_gradient_hist(netD),
                                global_step=epoch * len(dataloader) + i)

                # Output training stats
                if i % 50 == 0:
                    print(f'[{epoch+1}/{num_epochs}] [{i}/{len(dataloader)}] '
                        f'D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f} '
                        f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}')

                writer.add_scalar('D(x)', D_x, epoch * len(dataloader) + i)
                writer.add_scalar('D(G(z))', D_G_z1, epoch * len(dataloader) + i)


            
            if(epoch % 5 == 0):
            # Save the models
                torch.save(netG.state_dict(), f'generatorHPCWGANCLIPPINGPESI_{epoch+1}.pth')
                torch.save(netD.state_dict(), f'discriminatorHPCWGANCLIPPINGPESI_{epoch+1}.pth')

                # Save some samples
                # The code snippet provided is a Python script using PyTorch and TensorBoard to
                # generate and save fake images created by a Generative Adversarial Network (GAN)
                # model. 
                with torch.no_grad():
                    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
                    fixed_labels = torch.zeros(64, label_dim, image_size, image_size, device=device)
                    fake = netG(fixed_noise, fixed_labels)
                    vutils.save_image(fake.detach(), f'imagesHPCWGANCLIPPINGPESI_{epoch+1}.png', normalize=True)
                    
                    fake_image = fake[0]
                
                    fake_image = fake_image.cpu().numpy()
                    fake_image = np.transpose(fake_image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

                    # Add the image to TensorBoard
                    
                    writer.add_image('FAKEWGANCLIPPING Star', fake_image, dataformats='HWC')
            
            print ("FINE DEL TRAINING : ) ")
    
    # The above code snippet is checking if the value of `args.operation` is equal to the string
    # 'test'. If it is, then it calls the function `test()`. This is typically part of a larger
    # conditional block where different operations are performed based on the value of
    # `args.operation`.
    elif args.operation == 'test':
        test()
