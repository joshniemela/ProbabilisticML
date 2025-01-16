import torch
import torch.nn as nn
import numpy as np
signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  500



class SDE_DDPM(nn.Module):
    def __init__(self, network, sigma=25):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """

        super(SDE_DDPM, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze()))
        ).reshape(-1, 28 * 28)

        self.sigma = sigma

    def marginal_prob_std (self, t):
        return torch.sqrt((self.sigma**(2*t) - 1.) / 2. / np.log(self.sigma))  
    
    def diffusion_coeff (self, t):
        return self.sigma ** t
    
    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """
        eps = 0.001
        device = "cuda:0"
        t= torch.ones(shape, device=device)
        # Sample xT: Gaussian noise
        xt = torch.randn(shape).to(device) * self.marginal_prob_std(t)
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        with torch.no_grad ():
            for time_step in time_steps:
                batch_time_step = torch.ones(shape, device=device) * time_step
                g = self.diffusion_coeff(batch_time_step)
                mean_x = xt + (g**2) * self.network(xt, batch_time_step[:,0]) * step_size
                xt = mean_x + torch.sqrt(step_size) * g * torch.randn(shape).to(device)
        return xt

       # xt = xT
        #for t in range(self.T, 0, -1):
       #     noise = torch.randn_like(xT) if t > 1 else 0
       #     t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
       #     xt = self.reverse_diffusion(xt, t, noise)


    
    def loss(self, x0, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            model: A PyTorch model instance that represents a 
            time-dependent score-based model.
            x0: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand((x0.shape[0], 1), device=x0.device) * (1. - eps) + eps  
        z = torch.randn_like(x0)
        std = self.marginal_prob_std(random_t) # forward 
        perturbed_x = x0 + z * std
        score = self.network(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std + z)**2))
        return loss#/x0.shape[0]
    