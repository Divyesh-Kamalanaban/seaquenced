import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import json

# --- 1. Load the latent space data ---
# This assumes you have already run the autoencoder script
# which generated the latent space data.
try:
    df_latent = pd.read_csv("latent_space_autoencoder.csv")
    print("✅ Latent space data loaded successfully.")
except FileNotFoundError:
    print("Error: 'latent_space_autoencoder.csv' not found. Please run the autoencoder script first.")
    exit()

# Extract the (x, y) coordinates from the DataFrame
latent_points = df_latent[["x", "y"]].values

# --- 2. DBSCAN Clustering ---
# Apply DBSCAN clustering on the 2D latent space.
# The parameters (eps and min_samples) are crucial.
# You may need to tune these based on your data distribution.
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
dbscan = DBSCAN(eps=0.15, min_samples=4)
clusters = dbscan.fit_predict(latent_points)
df_latent['dbscan_cluster'] = clusters

# --- 3. Save Data to CSV File ---
# Convert the DataFrame to a CSV file.
df_latent.to_csv("latent_space_with_clusters.csv", index=False)
print("✅ DBSCAN results saved to 'latent_space_with_clusters.csv'")

# --- 4. Visualization of DBSCAN Clusters ---
plt.figure(figsize=(10, 8))
# Assign a different color for each cluster. 'viridis' is a good colormap.
scatter = plt.scatter(df_latent['x'], df_latent['y'], c=df_latent['dbscan_cluster'], cmap='viridis', s=10)
plt.title("2D Latent Space from Autoencoder with DBSCAN Clusters")
plt.xlabel("Latent Feature 1 (x)")
plt.ylabel("Latent Feature 2 (y)")
# Create a legend for the clusters
legend_labels = np.unique(df_latent['dbscan_cluster'])
legend_handles = scatter.legend_elements()[0]
plt.legend(legend_handles, legend_labels, title="Clusters")
plt.savefig("dbscan_clusters.png")
print("✅ Visualization of DBSCAN clusters saved to 'dbscan_clusters.png'")
plt.show()

# --- 5. Hierarchical Clustering and Dendrogram ---
print("\n--- Generating Dendrogram ---")
# Filter out noise points (cluster -1) for a cleaner dendrogram.
# Dendrograms are best at visualizing relationships within clusters, not noise.
clustered_points = df_latent[df_latent['dbscan_cluster'] != -1][['x', 'y']].values

if len(clustered_points) > 0:
    # Use Ward's method for linkage, which minimizes the variance of the clusters being merged.
    Z = linkage(clustered_points, method='ward')

    plt.figure(figsize=(15, 7))
    plt.title("Hierarchical Clustering Dendrogram (from DBSCAN clusters)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    dendrogram(Z,
               truncate_mode='lastp',  # show only the last p merged clusters
               p=20,  # show 20 merged clusters
               leaf_rotation=90.,
               leaf_font_size=8.,
               show_contracted=True)
    plt.savefig("dendrogram.png")
    print("✅ Dendrogram saved to 'dendrogram.png'")
    plt.show()
else:
    print("No clusters found by DBSCAN (all points labeled as noise). Dendrogram cannot be generated.")
    
# --- 6. Output Results ---
print("\n--- DBSCAN Analysis Complete ---")
num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)
print(f"Number of clusters found: {num_clusters}")
print(f"Number of unclustered sequences (potential novel taxa): {num_noise}")

# You can now use df_latent which has the 'dbscan_cluster' column for further analysis.
print("\nFirst 5 rows of the DataFrame with new cluster assignments:")
print(df_latent.head())