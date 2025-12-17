import matplotlib.pyplot as plt

sigma = [0, 3, 5, 7]

data = {
    "DinoV2":   [91.97, 82.21, 63.06, 40.16],
    "Ours(1K)":   [91.75, 84.03, 67.83, 46.90],
    "Ours(2K)":   [92.09, 85.37, 72.02, 53.66],
    "Ours(3K)":   [92.34, 85.53, 73.74, 55.18],
    "Ours(5K)":   [92.10, 85.48, 73.40, 55.15],
    "Ours(10K)":  [92.24, 85.14, 72.03, 53.68],
}

plt.figure(figsize=(4, 3))

for label, values in data.items():
    plt.plot(sigma, values, label=label)   


plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
for tick in plt.gca().get_xticklabels():
    tick.set_fontweight('bold')
for tick in plt.gca().get_yticklabels():
    tick.set_fontweight('bold')
    
plt.xlabel("Noise Level σ", fontsize=10, fontweight='bold')
plt.ylabel("Performance", fontsize=10, fontweight='bold')
plt.legend(prop={'size': 10, 'weight':'bold'})
plt.grid(axis='y')
plt.tight_layout()
plt.show()
