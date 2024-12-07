
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
    return samples

# analytically found that k=2 provides an upper bound with for a uniform proposal.
k_uf = 4 # upperbound on p(x) 

def rejection_sample_uf ():
    z0 = np.random.uniform(-3,3) # sample on q(x) 
    u0 = np.random.uniform(0, k_uf) # k*q(z0) is constant

    if u0 > p(z0): #reject
        return None
    else:
        return z0


def q_norm(x):
    """
    mean = 0
    std = 1
    """
    return (1 /(np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x ** 2))


def rejection_sample_norm():
    """
    k = 10
    """
    z0 = np.random.normal(0,1)
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


# Sample from each distribution and compute
# expectation, variance of X^2

classes = ["10", "100", "1000"]

x = np.arange(len(classes))  # x positions for each class
width = 0.3  # Width of each bar


rsu = []  # uniform rejection samples
rsn = []  # Gaussian rejection samples
snis = []  # Gaussian importance samples (self-normalized)

rsu_std = []  # standard deviations for uniform rejection samples
rsn_std = []  # standard deviations for Gaussian rejection samples
snis_std = []  # standard deviations for importance samples

for i in [10, 100, 1000]:
    rsu_samples = np.array(Rejection_sampler_uni(i))**2
    rsn_samples = np.array(Rejection_sampler_norm(i))**2
    snis_samples = importance_sampler(i)**2

    # Compute mean and standard deviation for the samples
    rsu.append(np.mean(rsu_samples))
    rsn.append(np.mean(rsn_samples))
    snis.append(np.mean(snis_samples))

    rsu_std.append(np.std(rsu_samples))  # Standard deviation
    rsn_std.append(np.std(rsn_samples))  # Standard deviation
    snis_std.append(np.std(snis_samples))  # Standard deviation

fig, ax = plt.subplots()

# Create the bar chart with error bars
bars_rsu = ax.bar(x - width / 2, rsu, width, yerr=rsu_std, label='Uniform Rejection Sampler', capsize=5)
bars_rsn = ax.bar(x + width / 2, rsn, width, yerr=rsn_std, label='Gaussian Rejection Sampler', capsize=5)
bars_snis = ax.bar(x + width, snis, width, yerr=snis_std, label='Importance Sampler', capsize=5)

ax.set_xlabel('Classes')
ax.set_ylabel('Mean of X')
ax.set_title('Comparison of Sampling Methods with Standard Deviation')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add values above bars for clarity
for bars in [bars_rsu, bars_rsn, bars_snis]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom'
        )

# Display the plot
plt.tight_layout()
plt.show()
