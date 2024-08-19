import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('C:\\Users\\vikrant.singh\\Downloads\\123Configurations\\summary(1).csv')  # Replace 'path_to_your_datafile.csv' with the actual path to your data file

# Function to create the enhanced clustered bar charts
# Function to create enhanced clustered bar charts with white background and horizontal lines at y-ticks
def create_enhanced_clustered_bar_charts_with_lines(dataset_data, dataset_name):
    # Set the seaborn style to 'white' and increase font scale
    sns.set(style='white', font_scale=2)

    # Update 'model' column to have more descriptive names
    dataset_data_updated = dataset_data.replace({'model': {'NAS': 'Neural Network Function Space'}})

    # Prepare the data: we need to 'melt' the dataframe to have a long-form dataframe suitable for sns.barplot
    melted_data = dataset_data_updated.melt(id_vars=['model', 'scoring_func', 'loss_function_new'],
                                            value_vars=['testMRR'],
                                            var_name='Metric', value_name='Value')

    # Create a figure and a grid of subplots
    g = sns.catplot(x='scoring_func', y='Value', hue='loss_function_new', col='model', data=melted_data, kind='bar',
                    ci=None, palette='Set3', height=8, aspect=1)

    # Adjust layout, move the legend slightly to the left, and add horizontal grid lines for readability
    g.set_axis_labels("Scoring Function", "MRR").set_titles("{col_name}")
    g.legend.set_title("Loss Function", prop={'size': 16})
    g.legend.set_bbox_to_anchor((0.98, 0.5))  # Move the legend slightly to the left

    # Iterate through the axes to set the grid
    for i, ax in enumerate(g.axes.flatten()):
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='grey')
        ax.set_axisbelow(True)  # Ensure grid lines are behind the bars

        # Only show the left spine for the first plot
        if i != 0:
            ax.spines['left'].set_visible(False)

    plt.subplots_adjust(top=0.9)  # Adjust subplot to fit into the figure area


# Create the charts for both datasets
umls_data = data[data['dataset_dir'] == 'KGs/UMLS']
create_enhanced_clustered_bar_charts_with_lines(umls_data, "UMLS")

kinship_data = data[data['dataset_dir'] == 'KGs/KINSHIP']
create_enhanced_clustered_bar_charts_with_lines(kinship_data, "KINSHIP")

plt.show()