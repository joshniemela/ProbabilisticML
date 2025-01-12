
import os
import inspect
import json
from datetime import datetime
import torch
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import ExponentialMovingAverage




def train(
    model,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    device,
    ema=True,
    per_epoch_callback=None,
    json_filepath = None,
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
    per_epoch_callback: function
        Called at the end of every epoch
    json_filepath : str
        filename to stream epoch losses to
    """
    # file name with timestamp 
    
    # initial metrics dump to json file
    if json_filepath is not None:
        time_ref = datetime.now()
        dir_name = f"{json_filepath}_{time_ref.year}-{time_ref.month}-{time_ref.day}_{time_ref.hour}.{time_ref.minute}"
        os.makedirs(dir_name, exist_ok=True)  # Create the directory
        # Create the file path within the directory
        file_name_ts = os.path.join(dir_name, "data.json")
        frame = inspect.currentframe()
        args_info = inspect.getargvalues(frame)
        params = args_info.locals
        print(params.keys())
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

    for epoch in range(epochs):
        
        # Switch to train mode
        model.train()
        epoch_loss = 0.0 # Initialize epoch loss
        global_step_counter = 0

        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
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
        if json_filepath is not None:
            # Prepare the data to append
            epoch_data = {
                "epoch": epoch + 1,
                "average_loss": average_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            # Append data step-by-step to JSON file
            with open(file_name_ts, "r+") as json_file:
                metrics = json.load(json_file)  # Load existing data
                metrics[epoch+1] = epoch_data  # Add new epoch data
                json_file.seek(0)  # Go back to the beginning of the file
                json.dump(metrics, json_file, indent=4)  # Write updated data
                json_file.truncate()  # Remove any leftover data
            
        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model)
    

def reporter(model):
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
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.show()

