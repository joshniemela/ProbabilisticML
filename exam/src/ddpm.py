import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
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

        super(DDPM, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean = (
            1.0
            / torch.sqrt(self.alpha[t])
            * (
                xt
                - (self.beta[t])
                / torch.sqrt(1 - self.alpha_bar[t])
                * self.network(xt, t)
            )
        )
        std = torch.where(
            t > 0,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

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

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def elbo_simple(self, x0, t):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image
        t: torch.tensor
            Batch of timesteps

        Returns
        -------
        float
            ELBO value
        """

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        return -nn.MSELoss(reduction="mean")(epsilon, self.network(xt, t))

    def loss(self, x0, sampler="iid"):
        """
        Loss function. Just the negative of the ELBO.
        """
        assert sampler in ["iid", "lds"]

        k = x0.shape[0]
        # Sample time step t
        # Independent sampling of t
        if sampler == "iid":
            t = torch.randint(1, self.T, (k, 1)).to(x0.device)
        # Low discrepency sampler
        elif sampler == "lds":
            u0 = torch.rand(1)
            i = torch.arange(1, k + 1)
            t = (
                torch.round(torch.remainder(u0 + i / k, 1) * self.T)
                .int()
                .unsqueeze(dim=1)
                .to(x0.device)
            )

        return -self.elbo_simple(x0, t).mean()
    

# WARNING DOES NOT WORK WITH EMA (Resets to init values during training)
class ImportanceDDPM(DDPM):
    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2, k=10):
        """
        Initialize ImportanceDDPM, inheriting from DDPM.
        
        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        T: int
            The number of diffusion steps.
        beta_1: float
            beta_t value at t=1.
        beta_T: float
            beta_t value at t=T.
        k: int
            Number of previous losses saved for each t, used in importance sampling.
        """
        super().__init__(network, T, beta_1, beta_T)

        self.k = k
        # Track previous losses for each timestep
        self.prev_losses = torch.zeros((T, k), dtype=torch.float)

        # Track the number of samples for each timestep
        self.t_num_samples = {t: 0 for t in range(1, T)}

        # State bool for initializing
        self.initializing = True

    def update_prev_losses(self, t, new_loss):
        """
        Update the rolling history of losses for a specific timestep t.
        Update the num_samples accordingly

        Parameters
        ----------
        t: int
            The timestep index.
        new_loss: float
            The new loss value to add to the history.
        """
        new_loss
        # Roll the previous losses to make space for the new loss
        self.prev_losses[t] = torch.roll(self.prev_losses[t], -1)
        self.prev_losses[t, -1] = new_loss

        # Check if initialization is complete
        if self.initializing:
            # Update the number of samples for this timestep, capped at 10
            self.t_num_samples[t] = min(self.t_num_samples[t] + 1, self.k)
            #print(self.t_num_samples)
            # Update initialization bool if init complete 
            self.initializing = not all(self.t_num_samples[num] == self.k for num in self.t_num_samples)
            if not self.initializing:
                print("initialization done, will start importance sampling now")

      
        
     

                
    # Copy of loss method from DDPM (added update_prev_losses)
    def pre_loss(self, x0, sampler="iid"):
        """
        Loss function. Just the negative of the ELBO.
        """
        
        assert sampler in ["iid", "lds"]

        k = x0.shape[0]
        # Sample time step t
        # Independent sampling of t
        if sampler == "iid":
            t = torch.randint(1, self.T, (k, 1)).to(x0.device)
        # Low discrepency sampler
        elif sampler == "lds":
            u0 = torch.rand(1)
            i = torch.arange(1, k + 1)
            t = (
                torch.round(torch.remainder(u0 + i / k, 1) * self.T)
                .int()
                .unsqueeze(dim=1)
                .to(x0.device)
            )
        loss = -self.elbo_simple(x0, t).mean()
        for i in range(k):
            self.update_prev_losses(t[i].item(), loss.item())
        return loss

    def importance_loss(self, x0):
        """
        Compute the loss using importance sampling.

        Parameters
        ----------
        x0: torch.tensor
            Input batch of images.

        Returns
        -------
        float
            Computed loss value.
        """
        # Example pseudocode
        k = x0.shape[0]

        # Calculate importance weights from self.prev_losses
        with torch.no_grad():
            importance_weights = self.prev_losses.pow(2).mean(dim=1).sqrt()  # E[L_t^2]
            importance_weights = importance_weights / (importance_weights.sum() + 1e-8)  # Normalize (avoid zero division)
        
        # Sample timesteps based on importance weights
        t = torch.multinomial(importance_weights, k, replacement=True).unsqueeze(dim=1).to(x0.device)

        # Compute loss for sampled timesteps
        epsilon = torch.randn_like(x0)
        xt = self.forward_diffusion(x0, t, epsilon)
        predicted_epsilon = self.network(xt, t)
        loss = nn.MSELoss(reduction="mean")(epsilon, predicted_epsilon)

        # Update rolling history
        for i in range(k):
            self.update_prev_losses(t[i].item(), loss.item())
        
        return loss

    def loss(self, x0):
        """
        Override the loss function to use importance sampler after initialization

        Parameters
        ----------
        x0: torch.tensor
            Input batch of images.

        Returns
        -------
        float
            Computed loss value.
        """
        if self.initializing:
            return self.pre_loss(x0)
        else:
            return self.importance_loss(x0)