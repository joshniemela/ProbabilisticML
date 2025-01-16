
import os
import inspect
import json
from datetime import datetime
import torch
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from utils import ExponentialMovingAverage

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3
import torchvision.transforms as T

# Number of samples used  for inception and fid estimates
N=100 

def calculate_inception_score(samples, inception_model, device):
    transform = T.Compose([T.Resize(299), T.CenterCrop(299), T.Normalize((0.5,), (0.5,))])
    samples = transform(samples)
    with torch.no_grad():
        preds = torch.nn.functional.softmax(inception_model(samples), dim=-1)
    p_yx = preds.cpu().numpy()
    p_y = np.mean(p_yx, axis=0)
    kl_div = p_yx * (np.log(p_yx + 1e-6) - np.log(p_y + 1e-6))
    return np.exp(np.mean(np.sum(kl_div, axis=1)))



def train(
    model,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    device,
    ema=True,
    dropout=None,
    per_epoch_callback=None,
    json_filepath = None,
    lds = False
):
    """
    Training loop

    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    dataloader: utils.DataLoader
        Pytorch dataloader
    epochs: int
        Number of epochs to train
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    dropout: float or None
        set to float to use conditional DDPM eg 0.2
    per_epoch_callback: (function, int) or None
        Call function on the current model, some number of times 
    json_filepath : str
        filename to stream epoch losses to
    """
    # file name with timestamp 

    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # initial metrics dump to json file
    if json_filepath is not None:
        time_ref = datetime.now()
        dir_name = f"{json_filepath}/{time_ref.year}-{time_ref.month}-{time_ref.day}_{time_ref.hour}.{time_ref.minute}"
        os.makedirs(dir_name, exist_ok=True)  # Create the directory
        # Create the file path within the directory
        file_name_ts = os.path.join(dir_name, "data.json")
        frame = inspect.currentframe()
        args_info = inspect.getargvalues(frame)
        params = args_info.locals
        metrics = {
            name: type(params[name]).__name__ if isinstance(params[name], object) else params[name]
            for name in params.keys()
        }
        with open(file_name_ts, "w") as json_file:
            json.dump(metrics, json_file, indent=4)

    num_batches = len(dataloader)
    # Setup progress bar
    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(
            model, device=device, decay=1.0 - ema_alpha
        )

    fid_scores = []
    inception_scores = []

    for epoch in range(epochs):
        
        # Switch to train mode
        model.train()
        epoch_loss = 0.0 # Initialize epoch loss
        global_step_counter = 0

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()

            if dropout: # use conditional DDPM
                y = y.to(device)
                # mask out some of the data with probability dropout
                if dropout > 0:
                    mask = torch.rand(y.shape[0], device=device) < dropout
                    # We mask values to 10 since 0...9 are the MNIST classes
                    y[mask] = 10
                loss = model.loss(x,y)
            elif lds:
                loss = model.loss(x, "lds")
            else:
                loss = model.loss(x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}",
            )
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter % ema_steps == 0:
                    ema_model.update_parameters(model)
        

        average_loss = epoch_loss / num_batches
        
        # Compute FID and Inception Score
        with torch.no_grad():
            # Generate samples from the model
                # Generate samples from the model
            samples = model.sample((N, 28 * 28)).to(device)

            # Reshape and normalize samples to match FID input requirements
            samples = samples.view(-1, 1, 28, 28)  # Reshape to (batch_size, channels, height, width)
            samples = (samples + 1) / 2  # Normalize to [0, 1]
            samples = samples.repeat(1, 3, 1, 1).to(device)   # Convert to 3 channels (batch_size, 3, 28, 28)

            # Update FID 
            fid.update(samples, real=False)

            # Update FID with real samples
            real_samples = dataloader.dataset.data.unsqueeze(1).float()[:N,:,:,:]  # MNIST real images
            real_samples = real_samples.repeat(1, 3, 1, 1).to(device) 
            fid.update(real_samples, real=True)

            # Compute FID score
            fid_score = fid.compute()
            fid_scores.append(fid_score.item())
            fid.reset()

            # Compute Inception Score
            inception_score = calculate_inception_score(samples, inception_model, device)
            inception_scores.append(inception_score)


        if json_filepath is not None:
            # Prepare the data to append
            epoch_data = {
                "epoch": epoch + 1,
                "average_loss": average_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "FID": fid_score.item(),
                "Inception Score": inception_score.item()
            }
            # Append data step-by-step to JSON file
            with open(file_name_ts, "r+") as json_file:
                metrics = json.load(json_file)  # Load existing data
                metrics[epoch+1] = epoch_data  # Add new epoch data
                json_file.seek(0)  # Go back to the beginning of the file
                json.dump(metrics, json_file, indent=4)  # Write updated data
                json_file.truncate()  # Remove any leftover data
        if per_epoch_callback and (epoch+1) % int(epochs/per_epoch_callback[1]) == 0:
            print("Callback")
            per_epoch_callback[0](ema_model.module if ema else model, epoch+1, dir_name)

# Modified to save image to file
def reporter(model, epoch, file_path):
    """Callback function used for plotting images during training"""

    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = 10
        samples = model.sample((nsamples, 28 * 28)).cpu()

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)

        # Plot and save
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.axis("off")
        plt.title(f"epoch: {epoch}")
        plt.savefig(file_path + f"/epoch{epoch}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


def cond_reporter(model, epoch, file_path):
    """Callback function used for plotting images during training"""

    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = 10
        # sample 0 to 10
        c = torch.randint(0, 10, (nsamples, ))

        samples = model.sample((nsamples, 28 * 28), c=c).cpu()

        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)

        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.axis("off")
        plt.title(f"epoch: {epoch}")
        plt.savefig(file_path + f"/epoch{epoch}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

