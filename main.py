# References: 
# https://discuss.pytorch.org/t/artifact-in-gan-generated-results/47372/2

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.data.data_loader import get_data_loaders
from src.utils.utils import get_device
from config import learning_rate, batch_size, num_epochs, save_interval, l1_lambda, id_loss_lambda

root = "./"

# Set device
device = get_device()

# Initialize Models
G_AtoB = Generator().to(device)
G_BtoA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Loss and Optimizers
lr = learning_rate
L1 = nn.L1Loss()
mse = nn.MSELoss()
optimizer_g = optim.Adam(list(G_AtoB.parameters()) + list(G_BtoA.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(0.5, 0.999))

# Data Loaders
loader_A, loader_B = get_data_loaders(os.path.join(root, 'DomainA'), os.path.join(root, 'DomainB'), batch_size, train="True")

# Training Loop
thresh_epoch = 100
for epoch in range(num_epochs):
    generator_loss_epoch = 0
    discriminator_loss_epoch = 0

    G_AtoB.train()
    G_BtoA.train()
    D_A.train()
    D_B.train()
    for data_A, data_B in tqdm(zip(loader_A, loader_B)):
        real_A, _ = data_A
        real_B, _ = data_B

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Zero the parameter gradients
        optimizer_d.zero_grad()

        # Discriminator Update
        fake_B = G_AtoB(real_A).detach()
        fake_A = G_BtoA(real_B).detach()

        d_A_loss = ((mse(D_A(real_A), torch.ones_like(D_A(real_A))) + mse(D_A(fake_A), torch.zeros_like(D_A(fake_A)))))
        d_B_loss = ((mse(D_B(real_B), torch.ones_like(D_B(real_B))) + mse(D_B(fake_B), torch.zeros_like(D_B(fake_B)))))

        d_loss =(d_A_loss + d_B_loss)/2
        discriminator_loss_epoch += d_loss.item()

        d_loss.backward()
        optimizer_d.step()

        # Generator Update
        optimizer_g.zero_grad()

        fake_B = G_AtoB(real_A)
        fake_A = G_BtoA(real_B)

        # cycle consistency loss
        cycle_A = G_BtoA(fake_B)
        cycle_B = G_AtoB(fake_A)

        # identity loss
        id_B = torch.sum(torch.abs(G_AtoB(real_B) - real_B))
        id_A = torch.sum(torch.abs(G_BtoA(real_A) - real_A))

        lamb = l1_lambda
        g_loss = mse(D_A(fake_A), torch.ones_like(D_A(fake_A))) + mse(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                 lamb * (L1(cycle_A, real_A) + L1(cycle_B, real_B)) + id_loss_lambda * (id_B  + id_A)
        generator_loss_epoch += g_loss.item()

        g_loss.backward()
        optimizer_g.step()


    if epoch >= thresh_epoch:
        # Calculate decayed learning rate
        decay_factor = (200 - epoch) / 100  # This linearly decreases from 1.0 to 0.0
        adjusted_lr = lr * decay_factor

        # Apply decayed learning rate to the optimizer
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = adjusted_lr
        for param_group in optimizer_g.param_groups:
            param_group['lr'] = adjusted_lr



    print(f'Epoch [{epoch+1}] | Generator Loss: {generator_loss_epoch:.4f} | Discriminator Loss: {discriminator_loss_epoch:.4f}')

    # Save models
    if (epoch + 1) % save_interval == 0:
        if not os.path.isdir(os.path.join(root, "state_dict")):
            os.mkdir(os.path.join(root, "state_dict"))
        if not os.path.isdir(os.path.join(root, "state_dict", "G_AtoB")):
            os.mkdir(os.path.isdir(os.path.join(root, "state_dict", "G_AtoB")))
        if not os.path.isdir(os.path.join(root, "state_dict", "G_BtoA")):
            os.mkdir(os.path.isdir(os.path.join(root, "state_dict", "G_BtoA")))
        if not os.path.isdir(os.path.join(root, "state_dict", "D_A")):
            os.mkdir(os.path.isdir(os.path.join(root, "state_dict", "D_A")))
        if not os.path.isdir(os.path.join(root, "state_dict", "D_B")):
            os.mkdir(os.path.isdir(os.path.join(root, "state_dict", "D_B")))

        torch.save(G_AtoB.state_dict(), os.path.join(root, "state_dict", "G_AtoB",  f'G_AtoB_epoch_{epoch+1}.pth'))
        torch.save(G_BtoA.state_dict(), os.path.join(root, "state_dict", "G_BtoA",  f'G_BtoA_epoch_{epoch+1}.pth'))

        torch.save(D_A.state_dict(), os.path.join(root, "state_dict", "D_A",  f'D_A_epoch_{epoch+1}.pth'))
        torch.save(D_B.state_dict(), os.path.join(root, "state_dict", "D_B",  f'D_B_epoch_{epoch+1}.pth'))
        print(f'Saved generators and discriminators at epoch {epoch+1}')

