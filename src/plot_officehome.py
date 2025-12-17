import matplotlib.pyplot as plt

sigma = [0, 3, 5, 7]

Art = {
    "ViT": {
        "base":  [80.25, 64.40, 44.03, 22.63],
        "indra": [79.63, 65.02, 43.62, 27.57],
    },
    "ConvNeXt": {
        "base":  [89.71, 62.76, 27.98, 12.14],
        "indra": [87.86, 59.88, 28.81, 14.20],
    },
    "DINOv2": {
        "base":  [87.65, 73.05, 46.91, 27.78],
        "indra": [87.04, 70.99, 47.53, 27.37],
    },
}
 
Clipart = {
    "ViT": {
        "base":  [73.20, 50.40, 28.64, 15.23],
        "indra": [69.76, 54.98, 33.10, 18.21],
    },
    "ConvNeXt": {
        "base":  [83.62, 54.07, 20.85, 09.74],
        "indra": [82.70, 57.85, 25.09, 11.34],
    },
    "DINOv2": {
        "base":  [88.43, 75.14, 51.09, 31.04],
        "indra": [87.29, 76.63, 54.75, 33.56],
    },
}


Product = {
    "ViT": {
        "base":  [92.34, 80.74, 61.15, 35.25],
        "indra": [89.75, 81.53, 64.08, 40.77],
    },
    "ConvNeXt": {
        "base":  [96.62, 84.91, 44.26, 19.37],
        "indra": [96.73, 85.92, 45.61, 22.18],
    },
    "DINOv2": {
        "base":  [96.73, 93.24, 83.33, 60.70],
        "indra": [96.40, 92.79, 84.46, 60.59],
    },
}

Real = {
    "ViT": {
        "base":  [89.22, 82.11, 60.09, 35.32],
        "indra": [87.16, 83.49, 63.65, 40.48],
    },
    "ConvNeXt": {
        "base":  [93.46, 82.11, 38.30, 17.78],
        "indra": [93.35, 84.63, 40.71, 19.61],
    },
    "DINOv2": {
        "base":  [92.78, 87.39, 71.44, 48.51],
        "indra": [92.89, 88.53, 73.17, 49.89],
    },
}


styles = {
    "ViT":      {"color": "C6", "marker": "o"},
    "ConvNeXt": {"color": "C1", "marker": "s"},
    "DINOv2":   {"color": "C2", "marker": "^"},
}

def plot_performance(data):

    plt.figure(figsize=(4, 3))

    for model, v in data.items():
        delta = [i - b for i, b in zip(v["indra"], v["base"])]
        plt.plot(
            sigma,
            delta,
            # marker=styles[model]["marker"],
            label=model,
            linewidth=2.5
        )

    plt.yticks([-2, 0, 2, 4])

    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontweight('bold')
    for tick in plt.gca().get_yticklabels():
        tick.set_fontweight('bold')
        
    plt.axhline(0, linestyle="--", linewidth=1.5, color="black", alpha=0.7)  
    plt.xlabel("Noise Level σ", fontsize=10, fontweight='bold')
    plt.ylabel("Performance Gain over Base", fontsize=10, fontweight='bold')
    plt.legend(prop={'size': 10, 'weight':'bold'})
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



plot_performance(Real)  # Art, Product, Clipart, Real