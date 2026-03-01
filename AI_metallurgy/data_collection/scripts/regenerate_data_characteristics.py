"""Regenerate data_characteristics.png with English-only labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(figsize=(10, 6))

categories = ['Experimental Only', 'Experimental + Computed']
feature_counts = [40, 35]
best_r2 = [0.67, 0.20]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, feature_counts, width, label='Feature Count', color='#6BAED6', alpha=0.8)
ax1.set_xlabel('Data Type', fontsize=12)
ax1.set_ylabel('Feature Count', color='#6BAED6', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#6BAED6')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, best_r2, width, label='Best R²', color='#FC8D62', alpha=0.8)
ax2.set_ylabel('Best R² Score', color='#FC8D62', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#FC8D62')
ax2.set_ylim(0, 0.8)

fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=11,
           framealpha=0.9)
plt.title('Data Characteristics and Best Model Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/nishioka/LUH/AI_metallurgy/data_collection/figures/data_characteristics.png',
            dpi=150, bbox_inches='tight')
print("Figure saved successfully.")
