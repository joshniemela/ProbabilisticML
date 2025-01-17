import torch
import torch.nn as nn

from ddpm import DDPM

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
        self.t_num_samples = {t: 0 for t in range(0, T)}

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
            t = torch.randint(0, self.T, (k, 1)).to(x0.device)
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
        k = x0.shape[0]

        # Calculate importance weights from self.prev_losses
        with torch.no_grad():
            importance_weights = self.prev_losses.pow(2).mean(dim=1).sqrt()  # E[L_t^2]
            if torch.all(importance_weights == 0):  # Handle uninitialized prev_losses
                importance_weights = torch.ones_like(importance_weights) / len(importance_weights)
            else:
                importance_weights = importance_weights / (importance_weights.sum() + 1e-8)  # Normalize

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

    def loss(self, x0, sampler= "iid"):
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
            return self.pre_loss(x0, sampler)
        else:
            return self.importance_loss(x0)
            