import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Your Data ---
# Data from "3.2 Training using averaged weights" (image_b27285.png)
avg_weights_acc = [94.75, 95.10, 95.05, 94.70, 95.30]

# Data from "Weight exchange count - 2" (image_b27282.png)
exchange_x2_acc = [95.35, 94.35, 93.40, 93.15, 94.95]

# Labels for the X-axis
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

# --- 2. Create a Pandas DataFrame ---
# This is the easiest way to plot grouped bars
df_data = {
    'Averaged Weights': avg_weights_acc,
    'Weight Exchange (Count 2)': exchange_x2_acc
}
df = pd.DataFrame(df_data, index=model_names)

# --- 3. Plot the Grouped Bar Chart ---
ax = df.plot(kind='bar', 
             figsize=(12, 7), 
             width=0.4) # Width of the bar groups

ax.set_title('Testing Accuracy: Averaged Weights vs. Weight Exchange (Count 2)', fontsize=16)
ax.set_ylabel('Testing Accuracy (%)')
ax.set_xlabel('Model')

# Set the Y-axis limit to zoom in on the differences
ax.set_ylim(90, 97) 

# Rotate x-axis labels to be horizontal
plt.xticks(rotation=0)

plt.legend(title='Training Method')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
plt.savefig('avg_vs_exchange_x2.png')
plt.show()