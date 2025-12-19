'''
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
    plt.savefig("./figs/radar.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    
radar_plot_with_clip_normalization(data, labels)
'''

import numpy as np
import matplotlib.pyplot as plt

labels = [
    "Top-5 T→I", "Top-5 I→T",
    "Top-10 T→I", "Top-10 I→T",
    "Top-30 T→I", "Top-30 I→T",
    "Top-50 T→I", "Top-50 I→T",
]


data_MSCOCO = {
    "Original":     [0.486, 0.491, 0.970, 0.981, 2.912, 2.927, 4.853, 4.874],     # ViT+Roberta
    "CCA":          [0.513,	0.519, 1.014, 1.028, 2.999,	3.019, 4.976, 5.016],     # ViT+Roberta
    "Linear Projection":       [0.460, 0.517, 0.963, 1.052, 2.845, 3.183, 4.890, 5.293],
    "Indra":        [1.048, 0.880, 2.065, 1.749, 5.970, 5.149, 9.702, 8.446],      # ViT+Roberta
    "CLIP":         [1.420, 1.381, 2.734, 2.661, 7.634, 7.470, 12.212, 11.986]
}

data_NOCAPS = {
    "Original":     [0.484, 0.483, 0.966, 0.963, 2.891, 2.886, 4.814, 4.805],       # ViT+Roberta
    "CCA":          [0.500, 0.489, 0.983, 0.979, 2.889, 2.948, 4.787, 4.915],       # ViT+Roberta
    "Linear Projection":       [0.496,	0.524, 1.058, 1.042, 3.037,	3.099, 4.938, 5.147],
    "Indra":        [0.924, 0.727, 1.792, 1.419, 5.014, 4.102, 8.011, 6.700],        # ViT+Roberta
    "CLIP":         [1.357, 1.325, 2.556, 2.499, 6.860, 6.717, 10.795, 10.584]
}

colors = {
    "CLIP":   "C9",   
    "Original":   "C1",   
    "Linear Projection":"C2",
    "CCA":"C6",
    "Indra":  "C0",    
}

def radar_plot_with_clip_normalization(
    data_left, 
    data_right, 
    labels, 
    clip_key="CLIP"
):
    
    def normalize_with_clip(data):
        clip_values = np.array(data[clip_key])
        data_norm = {}
        for name, values in data.items():
            values = np.array(values)
            data_norm[name] = (values / clip_values).tolist()
        return data_norm
    
    data_norm_left = normalize_with_clip(data_left)
    data_norm_right = normalize_with_clip(data_right)

    
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 3.5),
        subplot_kw=dict(polar=True)
    )
    
    ax_left, ax_right = axes
    
    legend_handles = []
    legend_labels = []
    
    
    for name, values in data_norm_left.items():
        values_plot = values + values[:1]
        line, = ax_left.plot(
            angles,
            values_plot,
            color=colors[name],
            linewidth=2,
            label=name
        )
        ax_left.fill(angles, values_plot, color=colors[name], alpha=0.1)

        if name not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(name)

    for name, values in data_norm_right.items():
        values_plot = values + values[:1]
        ax_right.plot(
            angles,
            values_plot,
            color=colors[name],
            linewidth=2
        )
        ax_right.fill(angles, values_plot, color=colors[name], alpha=0.1)
    
    
    for ax in [ax_left, ax_right]:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])

        ax.set_rlabel_position(180)
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])

        r_label = ax.get_rmax() * 1.10 
        for angle, label in zip(angles[:-1], labels):
            angle_deg = np.degrees(angle) % 360
            rotation = 0
            if abs(angle_deg - 0) < 10:
                rotation = -90
            elif abs(angle_deg - 180) < 10:
                rotation = 90
            elif abs(angle_deg - 90) < 10 or abs(angle_deg - 270) < 10:
                rotation = 0
            elif 0 < angle_deg < 90:
                rotation = -45
            elif 90 < angle_deg < 180:
                rotation = 45
            elif 180 < angle_deg < 270:
                rotation = 135
            elif 270 < angle_deg < 360:
                rotation = -135

            ax.text(
                angle,
                r_label,
                label,
                size=10,
                fontweight="bold",
                rotation=rotation,
                rotation_mode="anchor",
                ha="center",
                va="center",
            )

        for label in ax.get_xticklabels():
            label.set_fontsize(10)      
            label.set_fontweight('bold')    
        
    
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        prop={"weight": "bold", "size": 10}
    )

    plt.subplots_adjust(wspace=-0.45)
    plt.tight_layout()
    plt.savefig("./figs/radar.pdf", format="pdf", dpi=600, bbox_inches="tight")
    # plt.show()
    
radar_plot_with_clip_normalization(data_MSCOCO, data_NOCAPS, labels)