import numpy as np
import matplotlib.pyplot as plt

labels = [
    "Top-5 T→I", "Top-5 I→T",
    "Top-10 T→I", "Top-10 I→T",
    "Top-30 T→I", "Top-30 I→T",
    "Top-50 T→I", "Top-50 I→T",
]

data = {
    "CLIP":         [1.420, 1.381, 2.734, 2.661, 7.634, 7.470, 12.212, 11.986],
    "Original":     [0.482, 0.483, 0.967, 0.966, 2.911, 2.905, 4.863, 4.846],
    "LinPro":       [0.460, 0.517, 0.963, 1.052, 2.845, 3.183, 4.890, 5.293],
    "CCA":          [0.501, 0.498, 0.992, 1.001, 2.974, 3.004, 4.954, 5.001],
    "Indra":        [0.663, 0.832, 1.303, 1.613, 3.787, 4.426, 6.199, 7.036],
}

colors = {
    "CLIP":   "C9",   
    "Original":   "C1",   
    "LinPro":"C2",
    "CCA":"C6",
    "Indra":  "C0",     
}

def radar_plot_with_clip_normalization(data, labels, clip_key="CLIP"):
    clip_values = np.array(data[clip_key])

    data_norm = {}
    for name, values in data.items():
        values = np.array(values)
        data_norm[name] = (values / clip_values).tolist()

    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(polar=True))

    for name, values in data_norm.items():
        values = values + values[:1]
        ax.plot(angles, values, color=colors[name], linewidth=2, label=name)
        ax.fill(angles, values, color=colors[name], alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels,)
    ax.set_rlabel_position(180)

    for label in ax.get_xticklabels():
        label.set_fontsize(10)          
        label.set_fontweight('bold')    
    
    ax.tick_params(axis='x', pad=10)
       
    ax.set_ylim(0, 1.05)

    ax.tick_params(labelsize=10)

    ax.legend(loc="upper left", bbox_to_anchor=(-0.25, 0.85), prop={'weight': 'bold', 'size': 10})
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.show()
    
radar_plot_with_clip_normalization(data, labels)