import torch
import torch.nn as nn
import numpy as np
signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  500


def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones((batch_size, 1,1,1), device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in time_steps:      
      batch_time_step = torch.ones((batch_size, 1,1,1), device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2) * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

class SDE_DDPM(nn.Module):
    def __init__(self, network, T=100, sigma=25):
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

        # Total number of time steps
        self.T = T
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
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
    