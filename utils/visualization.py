import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

# def plot_umap(umap_emb, cell_types, study_ids, color_by, marker_by, result_dir, timestamp, title):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     color_field = study_ids if color_by == 'study_ids' else cell_types
#     marker_field = study_ids if marker_by == 'study_ids' else cell_types
#     unique_colors = np.unique(color_field)
#     unique_markers = np.unique(marker_field)

#     colors = cm.viridis(np.linspace(0, 1, len(unique_colors))) if len(unique_colors) > 8 else cm.Set1(np.linspace(0, 1, len(unique_colors)))
#     markers = ['o', '^', 's', 'D', '*', 'v', '<', '>', 'p', 'h', 'H', '8', 'P', 'X', 'd', '|', '_', '+', 'x']
#     num_markers = len(markers)

#     for cat in unique_colors:
#         for mk in unique_markers:
#             mask = (color_field == cat) & (marker_field == mk)
#             if np.sum(mask) > 0:
#                 marker_idx = list(unique_markers).index(mk) % num_markers
#                 ax.scatter(
#                     umap_emb[mask, 0],
#                     umap_emb[mask, 1],
#                     c=[colors[list(unique_colors).index(cat)]],
#                     marker=markers[marker_idx],
#                     s=50,
#                     alpha=0.6,
#                     edgecolors='white',
#                     linewidth=0.5
#                 )

#     color_legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label=str(cat), markerfacecolor=colors[i], markersize=8)
#         for i, cat in enumerate(unique_colors)
#     ]
#     leg1 = ax.legend(color_legend_elements, [str(cat) for cat in unique_colors],
#                      title="Color: " + ("Study IDs" if color_by == 'study_ids' else "Cell Types"),
#                      loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)

#     marker_legend_elements = [
#         Line2D([0], [0], marker=markers[i % num_markers], color='w', label=str(mk), markerfacecolor='gray', markersize=8)
#         for i, mk in enumerate(unique_markers)
#     ]
#     leg2 = ax.legend(marker_legend_elements, [str(mk) for mk in unique_markers],
#                      title="Marker: " + ("Study IDs" if marker_by == 'study_ids' else "Cell Types"),
#                      loc='upper left', bbox_to_anchor=(1.05, 0.5), fontsize=6)

#     ax.add_artist(leg1)
#     ax.set_title(title)
#     ax.set_xlabel('UMAP1')
#     ax.set_ylabel('UMAP2')

#     plt.tight_layout()
#     filename = f'umap_{color_by}_{timestamp}.png'
#     save_path = f"{result_dir}/{filename}"
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()

### Second version ###
# def plot_umap(umap_emb, cell_types, study_ids, color_by, marker_by, result_dir, timestamp, title):
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm
#     from matplotlib.lines import Line2D
#     import numpy as np

#     fig, ax = plt.subplots(figsize=(12, 8))
#     color_field = study_ids if color_by == 'study_ids' else cell_types
#     marker_field = study_ids if marker_by == 'study_ids' else cell_types

#     unique_colors = np.unique(color_field)
#     unique_markers = np.unique(marker_field)

#     color_map = plt.get_cmap('tab20', len(unique_colors))  # Discrete color map
#     markers = ['o', '^', 's', 'D', '*', 'v', '<', '>', 'p', 'h', 'H', '8', 'P', 'X', 'd', '|', '_', '+', 'x']
#     num_markers = len(markers)

#     for i, cat in enumerate(unique_colors):
#         for j, mk in enumerate(unique_markers):
#             mask = (color_field == cat) & (marker_field == mk)
#             if np.sum(mask) > 0:
#                 marker_idx = j % num_markers
#                 ax.scatter(
#                     umap_emb[mask, 0],
#                     umap_emb[mask, 1],
#                     c=[color_map(i)],
#                     marker=markers[marker_idx],
#                     s=50,
#                     alpha=0.7,
#                     edgecolors='white',
#                     linewidth=0.5
#                 )

#     # Categorical color legend
#     color_legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label=str(cat),
#                markerfacecolor=color_map(i), markersize=8)
#         for i, cat in enumerate(unique_colors)
#     ]
#     leg1 = ax.legend(color_legend_elements, [str(cat) for cat in unique_colors],
#                      title="Color: " + ("Study IDs" if color_by == 'study_ids' else "Cell Types"),
#                      loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)

#     # Marker legend
#     marker_legend_elements = [
#         Line2D([0], [0], marker=markers[i % num_markers], color='w', label=str(mk),
#                markerfacecolor='gray', markersize=8)
#         for i, mk in enumerate(unique_markers)
#     ]
#     leg2 = ax.legend(marker_legend_elements, [str(mk) for mk in unique_markers],
#                      title="Marker: " + ("Study IDs" if marker_by == 'study_ids' else "Cell Types"),
#                      loc='upper left', bbox_to_anchor=(1.05, 0.5), fontsize=6)

#     ax.add_artist(leg1)
#     ax.set_title(title)
#     ax.set_xlabel('UMAP1')
#     ax.set_ylabel('UMAP2')
#     plt.tight_layout()

#     filename = f'umap_{color_by}_{timestamp}.png'
#     save_path = os.path.join(result_dir, filename)
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()

### Version 3 ### 

def plot_umap(umap_emb, cell_types, study_ids, color_by, marker_by, result_dir, timestamp, title):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    import numpy as np
    import os

    def _plot(ax, color_field, marker_field, color_label, marker_label, title):
        unique_colors = np.unique(color_field)
        unique_markers = np.unique(marker_field)
        color_map = plt.get_cmap('tab20', len(unique_colors))
        markers = ['o', '^', 's', 'D', '*', 'v', '<', '>', 'p', 'h', 'H', '8', 'P', 'X', 'd', '|', '_', '+', 'x']
        num_markers = len(markers)

        for i, cat in enumerate(unique_colors):
            for j, mk in enumerate(unique_markers):
                mask = (color_field == cat) & (marker_field == mk)
                if np.sum(mask) > 0:
                    marker_idx = j % num_markers
                    ax.scatter(
                        umap_emb[mask, 0],
                        umap_emb[mask, 1],
                        c=[color_map(i)],
                        marker=markers[marker_idx],
                        s=50,
                        alpha=0.7,
                        edgecolors='white',
                        linewidth=0.5
                    )

        # Legends
        color_legend = [
            Line2D([0], [0], marker='o', color='w', label=str(cat),
                   markerfacecolor=color_map(i), markersize=8)
            for i, cat in enumerate(unique_colors)
        ]
        marker_legend = [
            Line2D([0], [0], marker=markers[i % num_markers], color='w', label=str(mk),
                   markerfacecolor='gray', markersize=8)
            for i, mk in enumerate(unique_markers)
        ]
        leg1 = ax.legend(color_legend, [str(c) for c in unique_colors],
                         title=f"Color: {color_label}", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)
        leg2 = ax.legend(marker_legend, [str(m) for m in unique_markers],
                         title=f"Marker: {marker_label}", loc='upper left', bbox_to_anchor=(1.05, 0.5), fontsize=6)
        ax.add_artist(leg1)
        ax.set_title(title)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')

    # Prepare color/marker combinations
    color_field_1 = study_ids if color_by == 'study_ids' else cell_types
    marker_field_1 = study_ids if marker_by == 'study_ids' else cell_types
    color_field_2 = cell_types if color_by == 'study_ids' else study_ids
    marker_field_2 = cell_types if marker_by == 'study_ids' else study_ids

    fig, axs = plt.subplots(1, 2, figsize=(22, 8))
    _plot(axs[0], color_field_1, marker_field_1, color_by, marker_by, f"{title} (Colored by {color_by})")
    _plot(axs[1], color_field_2, marker_field_2, marker_by, color_by, f"{title} (Colored by {marker_by})")

    plt.tight_layout()
    filename = f'umap_{color_by}_{timestamp}.png'
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_attention_heatmap(attn_map, tokens=None, title="Attention Map", save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_map.detach().cpu().numpy(),
                xticklabels=tokens if tokens else False,
                yticklabels=tokens if tokens else False,
                cmap='viridis')
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
