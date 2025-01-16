import torch
import numpy as np
 
from torchvision import datasets, transforms

from scorenet import ScoreNet
from cond_scorenet import ConditionalScoreNet
from train_utils import train, reporter, cond_reporter
from ddpm import DDPM
from importance_ddpm import ImportanceDDPM
from cond_ddpm import CondDDPM
from sde_ddpm import SDE_DDPM


# Parameters
T = 1000
num_reports = 5
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

#model = DDPM(mnist_unet, T=T).to(device)
base_model = DDPM(mnist_unet, T).to(device)

train(
    base_model,
    optimizer,
    scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=True,
    dropout=None,
    per_epoch_callback=[reporter,num_reports],
    json_filepath="exam/src/A/results/base_DDPM"
)

lds_model = DDPM(mnist_unet, T).to(device)
train(
    lds_model,
    optimizer,
    scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=True,
    dropout=None,
    per_epoch_callback=[reporter,num_reports],
    json_filepath="exam/src/A/results/lds_DDPM",
    lds=True
)

importance_model = ImportanceDDPM(mnist_unet, T).to(device) 
train(
    importance_model,
    optimizer,
    scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=False,
    dropout=None,
    per_epoch_callback=[reporter,num_reports],
    json_filepath="exam/src/A/results/ImportanceDDPM"
)

cond_mnist_unet = ConditionalScoreNet((lambda t: torch.ones(1).to(device)), num_classes=11)
cond_model = CondDDPM(cond_mnist_unet,T).to(device) 
# Construct optimizer
cond_optimizer = torch.optim.Adam(cond_model.parameters(), lr=learning_rate)
# Setup simple scheduler
cond_scheduler = torch.optim.lr_scheduler.ExponentialLR(cond_optimizer, 0.9999)

train(
    cond_model,
    cond_optimizer,
    cond_scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=True,
    dropout=0.2,
    per_epoch_callback=[cond_reporter,num_reports],
    json_filepath="exam/src/A/results/CondDDPM"
)

sigma =25


sde_mnist_unet = ScoreNet(lambda t: torch.sqrt((sigma**(2*t) - 1.) / 2. / np.log(sigma)) )
sde_model = SDE_DDPM(sde_mnist_unet, sigma).to(device)

train(
    sde_model,
    optimizer,
    scheduler,
    dataloader_train,
    epochs=epochs,
    device=device,
    ema=True,
    dropout=None,
    per_epoch_callback=[reporter,num_reports],
    json_filepath="exam/src/A/results/ImportanceDDPM"
)
