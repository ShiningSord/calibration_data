import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example dataset: results for two methods (method1 and method2)
data = {
    'Wanda': {'c4': [62.38,62.23,62.51,62.31,62.21,62.54,62.86,62.62,62.52,62.62,62.47,62.56,62.20,62.75,62.80,62.34,62.80,62.36,62.91,62.62], 'wiki': [61.16,61.01,61.39,60.90,61.35,60.93,60.66,60.89,61.01,61.08,60.97,60.71,60.86,61.06,60.73,61.04,61.02,61.11,61.48,61.14], 'slim': [61.91,62.01,62.45,62.51,62.13,62.05,62.07,62.18,62.34,62.25,62.06,62.24,62.75,62.54,62.28,62.62,62.59,62.50,62.29,62.31], 'dclm': [63.04,62.77,62.80,62.95,62.84,62.98,63.00,62.96,62.71,62.78,62.64,63.07,62.84,62.94,62.71,62.51,63.25,62.59,63.03,63.25]},
    'DSnoT': {'c4': [61.79,61.26,61.48,61.96,61.76,61.79,61.99,61.60,61.31,61.76,61.66,61.70,61.69,61.60,61.74,61.68,61.83,61.59,62.15,61.96], 'wiki': [60.79,60.25,60.48,60.39,60.76,60.54,60.58,60.16,60.43,60.83,60.25,60.34,60.52,60.48,60.08,60.61,60.18,60.42,61.00,60.57], 'slim': [60.76,60.88,61.31,61.31,61.43,61.19,61.46,61.30,61.19,61.14,60.90,61.43,61.50,61.43,61.00,61.18,61.04,61.28,61.30,61.05], 'dclm': [62.15,62.17,62.40,62.16,62.27,62.21,62.31,62.26,61.99,62.26,62.20,62.52,62.29,62.61,61.90,62.17,62.21,61.74,62.32,62.75]},
    
}

'''
# Create DataFrame for each method
df_c4 = pd.DataFrame({key: val['c4'] for key, val in data.items()})
df_wiki = pd.DataFrame({key: val['wiki'] for key, val in data.items()})
df_slim = pd.DataFrame({key: val['slim'] for key, val in data.items()})
df_dclm = pd.DataFrame({key: val['dclm'] for key, val in data.items()})

# Set plot style
sns.set(style="whitegrid")
# Initialize the figure with subplots (2 rows, 5 columns for 10 benchmarks)
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
# Flatten the axes array for easier indexing
axes = axes.flatten()
labels = {"c4": "C4", "wiki": "Wikipedia", "slim": "Slimpajama", "dclm": "DCLM"}
# Iterate over the benchmarks and create a violin plot for each one
for i, benchmark in enumerate(df_c4.columns):
    # Combine data from both methods
    combined_data = pd.DataFrame({
        'c4': df_c4[benchmark],
        'wiki': df_wiki[benchmark],
        'slim': df_slim[benchmark],
        'dclm': df_dclm[benchmark]
    })
    
    
    # Create separate violin plots for both methods on the same axis
    sns.violinplot(data=combined_data['c4'], ax=axes[i], color="blue", alpha=0.6, inner=None)
    sns.violinplot(data=combined_data['wiki'], ax=axes[i], color="red", alpha=0.6, inner=None)
    sns.violinplot(data=combined_data['slim'], ax=axes[i], color="lavender", alpha=0.6, inner=None)
    sns.violinplot(data=combined_data['dclm'], ax=axes[i], color="#add8e6", alpha=0.6, inner=None)

    
    # Set the title and labels for each sub-plot
    axes[i].set_title(benchmark)
    axes[i].set_ylabel('Accuracy (%)')
    axes[i].set_ylim(59, 64)
    axes[i].legend(labels.values())


# Adjust layout
plt.tight_layout()
'''


# Create a long-format DataFrame for plotting
def create_long_df(data):
    long_data = []
    for category, methods in data.items():
        for method, values in methods.items():
            long_data.extend([(category, method, value) for value in values])
    return pd.DataFrame(long_data, columns=['Category', 'Method', 'Accuracy'])

# Create long-format DataFrame
long_df = create_long_df(data)

# Set plot style
sns.set(style="whitegrid")

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 9))

# Define colors and labels for the legend
colors = {"c4": "blue", "wiki": "red", "slim": "lavender", "dclm": "#add8e6"}

# Plot each method with transparency
for method, color in colors.items():
    sns.violinplot(x='Category', y='Accuracy', data=long_df[long_df['Method'] == method], color=color, alpha=0.6, ax=ax, inner=None)

# Set titles and labels
#ax.set_title('Comparison of Methods: Wanda vs. DSnoT')
ax.set_ylabel('Performance')
ax.set_ylim(59.5, 64)

# Add legend manually
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()]
labels = ['C4', 'Wikipedia', 'Slimpajama', 'DCLM']
ax.legend(handles, labels, loc='upper right')

# Adjust layout
plt.tight_layout()


plt.savefig('violin.pdf')
