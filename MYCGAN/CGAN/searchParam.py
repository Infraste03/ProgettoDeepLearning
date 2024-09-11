"""
This script performs a grid search over various hyperparameters for training a Generative Adversarial Network (GAN) model. The primary components of the script include:

1. **Imports**: Necessary libraries and modules are imported for neural network operations, data handling, and image saving.

2. **Parameters**: Paths for the dataset and attribute files, as well as various hyperparameters like image size, number of epochs, and network dimensions, are specified.

3. **Initialization**:
   - **Device Setup**: Determines whether to use a GPU or CPU.
   - **Model Definition**: Initializes the Generator and Discriminator models and applies weight initialization.

4. **Loss Function**: Uses Binary Cross Entropy Loss (BCELoss) for training the GAN.

5. **Grid Search**:
   - **Hyperparameters**: Defines lists of learning rates, beta1 values for the Adam optimizer, and batch sizes to be tested.
   - **Training Loop**: For each combination of hyperparameters, updates the optimizers and creates a dataloader. It then trains the models for a specified number of epochs, printing and saving the generator and discriminator losses, as well as generating and saving sample images periodically.

6. **Model and Results Saving**: Saves the model checkpoints and generated images at regular intervals. At the end of the grid search, the best hyperparameters based on the lowest generator loss are recorded and saved to a file.

This approach helps in finding the optimal hyperparameter configuration by evaluating multiple combinations and monitoring performance during training.
"""

# The code is performing a grid search over different combinations of
# hyperparameters for training the Generative Adversarial Network (GAN) model.

#grid search: automatically explores various combinations of hyperparameters to find the best configuration.
#Continuous monitoring: the code prints and saves generator and discriminator losses for each epoch, as well as saving samples of images generated during training.
#Progressive saving: saves models and images periodically, which is useful to avoid data loss in the event of an interruption.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from dataloader import get_loader
from generator import Generator
from discriminator import Discriminator

# Parameters
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/DFSmall/minnie'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/attributeSmall.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Attractive','Big_Lips', 
                  'Black_Hair','Eyeglasses', 'Blond_Hair','5_o_Clock_Shadow','Arched_Eyebrows','Attractive']  

image_size = 64
nz = 100
ngf = 128
ndf = 128
num_epochs = 10
ngpu = 1
label_dim = len(selected_attrs)


############################################
if __name__ == '__main__':
    # Create the dataset and dataloader using the new method
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using device:", device)

    # Define the Generator and Discriminator
    

    netG = Generator(nz, ngf, label_dim).to(device)
    netD = Discriminator(ndf,label_dim).to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function
    criterion = nn.BCELoss()

    # Hyperparameter grid search
    """
    The script defines three lists of hyperparameters: learning rates, beta1 values for the Adam optimizer, and batch sizes.
    The grid search loop iterates over all combinations of these hyperparameters. 
     For each combination, it updates the optimizers for the generator and discriminator and creates a data loader using the specified batch size.
    """
    learning_rates = [0.0002, 0.0001, 0.00005]
    beta1_values = [0.5, 0.7, 0.9]
    batch_sizes = [32, 64, 128]

    best_loss = float('inf')
    best_params = {}

    # Grid Search Loop
    # The code is performing a grid search over different combinations of
    # hyperparameters for training the Generative Adversarial Network (GAN) model.
    for lr in learning_rates:
        for beta1 in beta1_values:
            for batch_size in batch_sizes:
                print(f"Training with learning rate: {lr}, beta1: {beta1}, batch_size: {batch_size}")
                
                # Update the optimizers
                optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
                optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

                # Create the dataloader
                dataloader = get_loader(dataroot, attr_path, selected_attrs, image_size=image_size, 
                                        batch_size=batch_size, mode='train', num_workers=2)

                # Initialize tracking for losses
                total_G_loss = 0
                total_D_loss = 0

                # Training Loop
                for epoch in range(num_epochs):
                    for i, data in enumerate(dataloader, 0):
                        # Update Discriminator
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
                        netG.zero_grad()
                        label.fill_(1)
                        output = netD(fake, labels).view(-1)
                        errG = criterion(output, label)
                        errG.backward()
                        D_G_z2 = output.mean().item()
                        optimizerG.step()



                        print(f'[{epoch+1}/{num_epochs}] '
                            f'[{i}/{len(dataloader)}] '
                            f'D_loss: {errD_real.item() + errD_fake.item()} '
                            f'G_loss: {errG.item()} '
                            f'D(x): {D_x} '
                            f'D(G(z)): {D_G_z1} / {D_G_z2}')



                        total_G_loss += errG.item()
                        total_D_loss += (errD_real.item() + errD_fake.item())

                    if epoch % 5 == 0:
                        # Save the models
                        torch.save(netG.state_dict(), f'generatorHPC10PARMA_lr{lr}_beta1{beta1}_bs{batch_size}_epoch{epoch+1}.pth')
                        torch.save(netD.state_dict(), f'discriminatorHPC10PARAM_lr{lr}_beta1{beta1}_bs{batch_size}_epoch{epoch+1}.pth')

                        # Save some samples
                        with torch.no_grad():
                            fixed_noise = torch.randn(64, nz, 1, 1, device=device)
                            fixed_labels = torch.zeros(64, label_dim, image_size, image_size, device=device)
                            fake = netG(fixed_noise, fixed_labels)
                            vutils.save_image(fake.detach(), f'imagesHPC10PARMA_lr{lr}_beta1{beta1}_bs{batch_size}_epoch{epoch+1}.png', normalize=True)

                            

                avg_G_loss = total_G_loss / len(dataloader)
                avg_D_loss = total_D_loss / len(dataloader)

                if avg_G_loss < best_loss:
                    best_loss = avg_G_loss
                    best_params = {'learning_rate': lr, 'beta1': beta1, 'batch_size': batch_size}

                print(f"Finished training with learning rate: {lr}, beta1: {beta1}, batch_size: {batch_size}")

    # Save the best parameters to a file
    with open('best_params.txt', 'w') as f:
        f.write(f"Best learning rate: {best_params['learning_rate']}\n")
        f.write(f"Best beta1: {best_params['beta1']}\n")
        f.write(f"Best batch size: {best_params['batch_size']}\n")
        f.write(f"Lowest generator loss: {best_loss}\n")

    print(f"Best parameters saved to 'best_params.txt'")
