
import numpy as np
import matplotlib.pyplot as plt

def p(x):
    """ """
    return np.exp(-(x**2)/2) * (np.sin(x)**2 + 3*np.cos(x)*(np.sin(7*x)**2) + 1)


def Generate_samples(n, sampler):
    """meta function used to make n samples"""
    samples = []
    while len(samples) != n:
        if (x := sampler()) != None:
            samples.append(x)
    return np.array(samples)

# analytically found that k=2 provides an upper bound with for a uniform proposal.



def rejection_sample_uf ():
    kq = 4 # upperbound on p(x) 
    z0 = np.random.uniform(-3,3) # sample on q(x) 
    u0 = np.random.uniform(0, kq) # k*q(z0) is constant

    if u0 > p(z0): #reject
        return None
    else:
        return z0


def q_norm(x):
    """
    mean = 0
    std = 1
    """
    return (1 /(np.sqrt(2 * np.pi))) * np.exp(-0.5 * x ** 2)


def rejection_sample_norm():
    """
    k = 4
    """
    z0 = np.random.normal(0,1)
    # ensure sample is in [-3,3]
    if not (-3 <= z0 <= 3): # proposal sample out of range
        return None 

    u0 = np.random.uniform(0, 10 * q_norm(z0))
    if u0 > p(z0): #reject
        return None
    else:
        return z0
    

def importance_sampler(num_samples, num_proposals=1000):
    """
    Self normalized importance sampler with Gaussian proposal.
    """
    # Find impotance distrbution
    proposals = np.random.normal(0,1,num_proposals)
    weights = np.clip(p(proposals)/q_norm(proposals), 0, None)
    total_weight = sum(weights)
    normalized_weights = weights/total_weight
   
    # sample from importance distribution
    resampled_indices = np.random.choice(
    np.arange(num_proposals), size=num_samples, p=normalized_weights
    )
    resampled_values = proposals[resampled_indices]
    
    return resampled_values

def Rejection_sampler_uni(n): return Generate_samples(n,rejection_sample_uf)

def Rejection_sampler_norm(n): return Generate_samples(n, rejection_sample_norm)


### estimating E[X^2] and plotting results

classes = ["10", "100", "1000"] # sample sizes
num_experiments = 1000

x = np.arange(len(classes))  # x positions for each class
width = 0.3  # Width of each bar
rsu = []  # uniform rejection samples
rsn = []  # Gaussian rejection samples
snis = []  # Gaussian importance samples (self-normalized)

rsu_std = []  # standard deviations for uniform rejection samples
rsn_std = []  # standard deviations for Gaussian rejection samples
snis_std = []  # standard deviations for importance samples

for i in [10, 100, 1000]:
    
    # sampling num_experiments
    rsu_samples = np.array([Rejection_sampler_uni(i)**2 for _ in range(num_experiments)])
    rsn_samples = np.array([Rejection_sampler_norm(i)**2 for _ in range(num_experiments)])
    snis_samples = np.array([importance_sampler(i)**2 for _ in range(num_experiments)])
    # compute X^2 estimates
    rsu_estimates = np.mean(rsu_samples, axis=1)
    rsn_estimates = np.mean(rsn_samples, axis=1) 
    snis_estimates = np.mean(snis_samples, axis=1) 
   
    # Compute mean and standard deviation of the estimates
    rsu.append(np.mean(rsu_estimates))
    rsn.append(np.mean(rsn_estimates))
    snis.append(np.mean(snis_estimates))

    rsu_std.append(np.std(rsu_samples)) 
    rsn_std.append(np.std(rsn_estimates))  
    snis_std.append(np.std(snis_estimates)) 

# Create the bar chart with error bars
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

# Adjust bar positions for better alignment
bars_rsu = ax.bar(
    x - width, rsu, width, yerr=rsu_std, label='Uniform Rejection Sampler', capsize=5, color='skyblue'
)
bars_rsn = ax.bar(
    x, rsn, width, yerr=rsn_std, label='Gaussian Rejection Sampler', capsize=5, color='lightgreen'
)
bars_snis = ax.bar(
    x + width, snis, width, yerr=snis_std, label='Importance Sampler', capsize=5, color='salmon'
)

# Add labels and titles
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('Mean of X', fontsize=12)
ax.set_title('Comparison of Sampling Methods with Standard Deviation', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=10)

# Add values above bars and show standard deviation
for bars, std_values in zip([bars_rsu, bars_rsn, bars_snis], [rsu_std, rsn_std, snis_std]):
    for bar, std in zip(bars, std_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9
        )
        # Add standard deviation to the right of the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + std + 0.02, f'(±{std:.2f})',
            ha='center', va='bottom', fontsize=9, color='darkblue'
        )

# Adjust layout for readability
plt.tight_layout()

# Display the plot
plt.show()


### Include NUTS sampler results (Hardcoded)

# Data for NUTS results (hardcoded)
nuts_means = [0.9122459888458252, 0.8607342839241028, 0.9301851391792297]
nuts_stds = [0.5354326367378235, 0.1312291920185089, 0.09108437597751617]

# Existing data
classes = ["10", "100", "1000"]  # sample sizes
num_experiments = 1000

# Positions for bars
x = np.arange(len(classes))  # x positions for each class
width = 0.2  # Width of each bar

# Create the bar chart with error bars
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

# Add bars for each sampling method
bars_rsu = ax.bar(
    x - 1.5 * width, rsu, width, yerr=rsu_std, label='Uniform Rejection Sampler', capsize=5, color='skyblue'
)
bars_rsn = ax.bar(
    x - 0.5 * width, rsn, width, yerr=rsn_std, label='Gaussian Rejection Sampler', capsize=5, color='lightgreen'
)
bars_snis = ax.bar(
    x + 0.5 * width, snis, width, yerr=snis_std, label='Importance Sampler', capsize=5, color='salmon'
)
bars_nuts = ax.bar(
    x + 1.5 * width, nuts_means, width, yerr=nuts_stds, label='NUTS Sampler', capsize=5, color='orange'
)

# Add labels and titles
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('Mean of X', fontsize=12)
ax.set_title('Comparison of Sampling Methods with Standard Deviation', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=10)

# Add values above bars and show standard deviation
for bars, std_values in zip([bars_rsu, bars_rsn, bars_snis, bars_nuts], [rsu_std, rsn_std, snis_std, nuts_stds]):
    for bar, std in zip(bars, std_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9
        )
        # Add standard deviation to the right of the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + std + 0.02, f'(±{std:.2f})',
            ha='center', va='bottom', fontsize=9, color='darkblue'
        )

# Adjust layout for readability
plt.tight_layout()

# Display the plot
plt.show()





