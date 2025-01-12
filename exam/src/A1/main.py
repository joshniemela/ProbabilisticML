import torch
from torchvision import datasets, transforms

from scorenet import ScoreNet
from train_utils import train, reporter
from ddpm import DDPM
from importance_ddpm import ImportanceDDPM

# Parameters
T = 1000
learning_rate = 1e-3
epochs = 100
batch_size = 256


# Rather than treating MNIST images as discrete objects, as done in Ho et al 2020,
# we here treat them as continuous input data, by dequantizing the pixel values (adding noise to the input data)
# Also note that we map the 0..255 pixel values to [-1, 1], and that we process the 28x28 pixel values as a flattened 784 tensor.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: x + torch.rand(x.shape) / 255
        ),  # Dequantize pixel values
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # Map from [0,1] -> [-1, -1]
        transforms.Lambda(lambda x: x.flatten()),
    ]
)

# Download and transform train dataset
dataloader_train = torch.utils.data.DataLoader(
    datasets.MNIST("./mnist_data", download=True, train=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Construct Unet
# The original ScoreNet expects a function with std for all the
# different noise levels, such that the output can be rescaled.
# Since we are predicting the noise (rather than the score), we
# ignore this rescaling and just set std=1 for all t.
mnist_unet = ScoreNet((lambda t: torch.ones(1).to(device)))

# Construct model
model = DDPM(mnist_unet, T=T).to(device)

# Construct optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup simple scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

# Call training loop
train(
    model,
    optimizer,
    scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=True,
    per_epoch_callback=reporter,
    json_filepath="DDPM"
)
