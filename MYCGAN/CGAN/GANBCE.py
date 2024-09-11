"""
This Python script is designed to train and test a Generative Adversarial Network (GAN) using the CelebA dataset.
The script uses PyTorch and various libraries to implement and manage the training process for both the Generator and Discriminator networks.

The script performs the following tasks:

1. **Imports Required Libraries:**
   - The necessary Python libraries and modules are imported, including PyTorch for deep learning, torchvision for image processing, and TensorBoard for logging.

2. **Sets Parameters:**
   - Defines various configuration parameters such as data paths, batch size, image size, and hyperparameters for the model.

3. **Initializes TensorBoard Writer:**
   - Sets up a TensorBoard SummaryWriter for logging training progress and visualizations.

4. **Command-Line Interface:**
   - Uses the `argparse` module to create a command-line interface, allowing the user to specify whether to run training or testing.

5. **Training Procedure:**
   - Creates data loaders for the training data.
   - Initializes the Generator and Discriminator models.
   - Applies weight initialization to the models.
   - Defines loss functions and optimizers.
   - Executes the training loop, including updating the Discriminator and Generator, logging gradients, and saving models and images at regular intervals.

6. **Testing Procedure:**
   - If specified, the script will run the testing phase by calling the `test()` function.

The script ensures that the models are saved periodically and visualizes some generated images using TensorBoard.
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
from dataloader import get_loader
from generator import Generator
from discriminator import Discriminator
from TestBCE import test
from torch.utils.tensorboard import SummaryWriter
from matplotlib.lines import Line2D
#dos2unix sbatch.sh

# Parameters

# The above code is a Python script that defines various configuration parameters for a machine
# learning project related to image processing.
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/images'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/list_attr_celeba.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Attractive','Big_Lips', 
                  'Black_Hair','Eyeglasses', 'Blond_Hair','5_o_Clock_Shadow','Arched_Eyebrows','Attractive']  
batch_size = 128
image_size = 64
nz = 100
ngf = 64
ndf = 64
num_epochs = 10 # number of epochs 5,10,15,50
lr = 0.0002 
beta1 = 0.5
ngpu = 1
label_dim = len(selected_attrs)

###################
writer = SummaryWriter('./runs/tensorcGANBCE')
##################

if __name__ == '__main__':
    
    # The above code is a Python script using the `argparse` module to create a command-line interface
    # for a script that trains and tests a model. It defines a parser with a description for the
    # script, adds an argument 'operation' with choices 'train' or 'test' to specify whether to
    # perform training or testing, and then parses the command-line arguments provided when running
    # the script. If the 'operation' argument is 'train', it will execute the code block under that
    # condition.
    parser = argparse.ArgumentParser(description="Script per addestrare e testare il modello")
    parser.add_argument('operation', choices=['train', 'test'], help="Specifica se eseguire il training o il testing")
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
        
        ##########################
        img_grid = vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu()
        img_grid = img_grid.numpy()
        img_grid = np.transpose(img_grid, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

        # Add the image to TensorBoard
        
        writer.add_image('Example Star', img_grid, dataformats='HWC')
        """
        The function `add_gradient_hist` calculates and visualizes the average gradients of the
        parameters in a neural network.
        
        :param net: The `net` parameter in the `add_gradient_hist` function is typically a neural
        network model. The function calculates and visualizes the average gradient values for the
        parameters of each layer in the network. It iterates through the named parameters of the
        network, calculates the average gradient for each parameter (weight
        :return: The function `add_gradient_hist(net)` returns a matplotlib figure object that
        displays a bar chart showing the average gradients of the parameters in the neural network
        `net`. The chart visualizes the gradient flow through different layers of the network.
        """
        
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
        
        ##########################
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
        
        
        
        ########################DIFFERENT TYPE OF WEIGHTS INITIALITION##########################
        """ 
        
        
            The function `weights_init` initializes weights and biases of Convolutional, Batch
                Normalization, and Linear layers using Xavier initialization and constant
                initialization.
                
                :param m: The `weights_init` function is used to initialize the weights of different
                types of layers in a neural network. The function takes a module `m` as input, which
                represents a layer in the neural network. Depending on the type of layer (Convolutional,
                Batch Normalization, or Linear)
                ###############################################
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
                        
        ###############################################          
            The function `weights_init` initializes weights and biases of different types of layers
                in a neural network using specific initialization methods.
                
                :param m: The function `weights_init` is used to initialize the weights of different
                types of layers in a neural network. The function takes a module `m` as input, which
                represents a layer in the neural network. Depending on the type of layer (Convolutional,
                Linear, BatchNorm, LayerNorm
                        
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
        
        
        ##################################################
                
                
        # The code is applying the function `weights_init` to the `netG` and `netD` models in Python.
        # This function is likely used to initialize the weights of the neural network models before
        # training.
        netG.apply(weights_init)
        netD.apply(weights_init)

        # Create the loss function and optimizers
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
        #optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
        #optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)
        #optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
        #optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

        # Training Loop 
        # The above code is implementing a training loop for a Generative Adversarial Network (GAN).
        
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                # Update Discriminator
                #The loss on both real and generated images is calculated, and the discriminator is updated.
                
                netD.zero_grad()
                real, labels = data
                batch_size = real.size(0)
                real = real.to(device)
                labels = torch.cat([labels] * (image_size // 64), dim=0).unsqueeze(2).unsqueeze(3).expand(-1, -1, image_size, image_size).to(device)
                label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                output = netD(real, labels).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise, labels)
                label.fill_(0)
                output = netD(fake.detach(), labels).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                optimizerD.step()

                # Update Generator
                # The loss of the generator is calculated, which tries to deceive the discriminator.
                netG.zero_grad()
                label.fill_(1)
                output = netD(fake, labels).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                
                ##############################

                # The code is using a Python library (TensorBoard) to add figures
                # representing the gradients of two neural networks (netG and netD) to a visualization
                # tool. I
                writer.add_figure('gradients',
                                add_gradient_hist(netG),
                                global_step=epoch * len(dataloader) + i)
                writer.add_figure('gradients',
                                add_gradient_hist(netD),
                                global_step=epoch * len(dataloader) + i)

                print(f'[{epoch+1}/{num_epochs}] '
                    f'[{i}/{len(dataloader)}] '
                    f'D_loss: {errD_real.item() + errD_fake.item()} '
                    f'G_loss: {errG.item()} '
                    f'D(x): {D_x} '
                    f'D(G(z)): {D_G_z1} / {D_G_z2}')
            
                writer.add_scalar('Discriminator Loss', errD_real.item() + errD_fake.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Generator Loss', errG.item(), epoch * len(dataloader) + i)
                writer.add_scalar('D(x)', D_x, epoch * len(dataloader) + i)
                writer.add_scalar('D(G(z))', D_G_z1, epoch * len(dataloader) + i)


            
            if(epoch % 2 == 0):
            # Save the models
                torch.save(netG.state_dict(), f'generatorBCEFINAL_{epoch+1}.pth')
                torch.save(netD.state_dict(), f'discriminatorBCEFINAL_{epoch+1}.pth')

                # Save some samples
                with torch.no_grad():
                    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
                    fixed_labels = torch.zeros(64, label_dim, image_size, image_size, device=device)
                    fake = netG(fixed_noise, fixed_labels)
                    vutils.save_image(fake.detach(), f'imagesBCEFINAL__{epoch+1}.png', normalize=True)
                    
                    fake_image = fake[0]
                
                    fake_image = fake_image.cpu().numpy()
                    fake_image = np.transpose(fake_image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

                    # Add the image to TensorBoard
                    
                    writer.add_image('Fake Img Star', fake_image, dataformats='HWC')
            
            print ("FINE DEL TRAINING : ) ")
        
   # The code snippet is checking if the value of `args.operation` is equal to the string 'test'. If
   # it is, then it calls the function `test()`.
    elif args.operation == 'test':
        test()
            
        
