from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os

# Ensure the directory exists
os.makedirs('results', exist_ok=True)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def binarize(x):
    return x > 0

def to_float(x):
    return x.float()



#  provided train/test loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       binarize,
                       to_float,
            ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       binarize,
                       to_float,
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc21 = nn.Linear(100, 2)  # Output size changed to 2
        self.fc22 = nn.Linear(100, 2)  # Output size changed to 2
        self.fc3 = nn.Linear(2, 100)   # Input size changed to 2
        self.fc4 = nn.Linear(100, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            # Collect latent vectors and corresponding labels
            latent_vectors.append(mu.cpu().numpy())
            labels.append(target.numpy())

            # Save reconstruction for the first batch
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           f'results/reconstruction_{epoch}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

    # Combine latent vectors and labels into a single array
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    print(latent_vectors)
    labels = np.concatenate(labels, axis=0)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_vectors[:, 0], y=latent_vectors[:, 1], hue=labels, palette="tab10", alpha=0.7
    )
   # plt.colorbar().set_label("Digit Class")
    plt.title(f"Latent Space of VAE (Epoch {epoch})")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(title="Digit Class")
    plt.savefig(f'results/latent_space_{epoch}.png')

def plot_latent_space(epoch, k=10):
    k = k + 2  # Adjust for the outer edge images to be removed
    """
    Plot a saved sample image with scales and axis titles.
    """
    # Create the grid of latent points
    x = torch.linspace(-3, 3, k, device=device)
    y = torch.linspace(-3, 3, k, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # Standard Normal distribution
    normal_dist = torch.distributions.normal.Normal(0, 1)

    # Flatten the grid for processing
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Map to CDF
    transformed_points = normal_dist.icdf((grid_points + 3) / 6)  # Normalize to [0,1]
    
    # Decode images from latent points
    decoded_images = []
    for z in transformed_points:
        with torch.no_grad():
            decoded = model.decode(z.unsqueeze(0))  # Decode single latent point
            # Reshape each decoded image (784-dimensional) into a 28x28 image
            decoded_images.append(decoded.view(1, 28, 28).cpu())

    # Ensure we have exactly k*k images, pad with black
    num_images = len(decoded_images)
    required_images = k * k
    if num_images < required_images:
        # Add black images to pad
        padding = required_images - num_images
        black_image = torch.zeros(1, 28, 28)  # Black image (all zeros)
        decoded_images.extend([black_image] * padding)

    # Stack the images into a tensor
    decoded_images = torch.stack(decoded_images)

    # Reshape to the grid form for displaying (k x k grid)
    images = decoded_images.view(k, k, 1, 28, 28)  # Shape: (k, k, 1, 28, 28)

    # Remove the outermost rows and columns
    inner_images = images[1:-1, 1:-1]

    # Reshape the inner_images tensor for the grid
    inner_images = inner_images.reshape(-1, 1, 28, 28)  # Use reshape instead of view

    # Optionally, save the image grid to a file with custom padding
    grid_image = make_grid(inner_images, nrow=k-2, padding=0, pad_value=0)  # Adjust padding and pad color
    save_image(grid_image, 'results/plz_shaved.png')

    # Now, load and display the saved image using matplotlib
    img = mpimg.imread('results/plz_shaved.png')  # Read the image

    # Display the image in a figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    ax.axis('on')

    # Change the ticks to reflect the new scale (-3, 3)
    ax.set_xticks([0, img.shape[1] // 2, img.shape[1]])  # Set the positions for the ticks
    ax.set_xticklabels([-3, 0, 3])  # Set the labels for the ticks
    ax.set_yticks([0, img.shape[0] // 2, img.shape[0]])  # Set the positions for the y-axis ticks
    ax.set_yticklabels([-3, 0, 3])  # Set the labels for the y-axis tick

    ax.set_xlabel('Latent Space Dimension 1 (z1)', fontsize=14)
    ax.set_ylabel('Latent Space Dimension 2 (z2)', fontsize=14)
    ax.set_title(f"p(x|z) for Latent Space Mapping (Epoch {epoch})", fontsize=16)
    plt.savefig(f'results/P_latent_space_{epoch}.png')


if __name__ == "__main__":
    
    plot_latent_space(0)
    for epoch in range(1,11): # args.epochs + 1
        train(epoch)
        test(epoch)
        plot_latent_space(epoch)

    
    
      


