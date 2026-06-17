import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# --- 1. Load the clustered data ---
# This script assumes you have run the cnn_autoencoder.py and the dbscan_analysis.py
# which generated the latent space with cluster assignments.
try:
    df_clustered = pd.read_csv("latent_space_with_clusters.csv")
    print("✅ Clustered data loaded successfully.")
except FileNotFoundError:
    print("Error: 'latent_space_with_clusters.csv' not found. Please run the CNN and DBSCAN scripts first.")
    exit()

# Filter out noise points, which are typically not assigned a cluster ID.
clusters_df = df_clustered[df_clustered['cluster'] != -1]

# --- 2. Function to find a representative sequence for each cluster ---
def find_representative_sequences(df):
    """
    Finds the most representative sequence for each cluster.
    This is the data point closest to the cluster centroid.
    """
    representative_sequences = {}
    
    unique_clusters = df['cluster'].unique()
    for cluster_id in unique_clusters:
        # Get all data points for the current cluster
        cluster_points = df[df['cluster'] == cluster_id]
        
        # Calculate the cluster centroid
        centroid = cluster_points[['x', 'y']].mean().values.reshape(1, -1)
        
        # Calculate Euclidean distances from all points to the centroid
        distances = euclidean_distances(cluster_points[['x', 'y']].values, centroid)
        
        # Find the index of the point closest to the centroid
        closest_point_index = np.argmin(distances)
        
        # Get the row for the representative sequence
        representative_row = cluster_points.iloc[closest_point_index]
        representative_sequences[cluster_id] = representative_row
        
    return representative_sequences

# --- 3. Analyze the composition of each cluster ---
print("\n--- Cluster Composition Analysis ---")
print("This analysis shows which original species labels are grouped into each cluster.")

unique_clusters = np.sort(clusters_df['cluster'].unique())
representative_sequences = find_representative_sequences(clusters_df)

for cluster_id in unique_clusters:
    # Get all data points for the current cluster
    cluster_points = clusters_df[clusters_df['cluster'] == cluster_id]
    
    # Get the unique original labels within this cluster
    unique_labels = cluster_points['label'].unique()
    num_sequences = len(cluster_points)
    
    # Print a summary for the cluster
    print(f"\nCluster ID: {cluster_id}")
    print(f"  Number of sequences: {num_sequences}")
    
    # Check if the cluster is homogeneous or heterogeneous
    if len(unique_labels) == 1:
        print(f"  This is a HOMOGENEOUS cluster.")
        print(f"  All sequences belong to the known species: '{unique_labels[0]}'")
        print("   This demonstrates the pipeline's ability to accurately group known species.")
    else:
        print(f"  This is a HETEROGENEOUS cluster.")
        print("  It contains a mix of known species, indicating potential new taxonomic relationships.")
    
    # Count the occurrences of each unique label within the cluster
    label_counts = cluster_points['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  - Contains {count} sequences from '{label}'")
        
    # Print the representative sequence for real-world use
    rep_sequence = representative_sequences[cluster_id]
    print(f"\n  Representative sequence for BLAST search (closest to centroid):")
    print(f"    - Original Label: {rep_sequence['label']}")
    print(f"    - Latent Coordinates: ({rep_sequence['x']:.4f}, {rep_sequence['y']:.4f})")


# --- 4. Rationale for finding new taxa ---
print("\n--- Rationale for Unsupervised Discovery ---")
print("The primary goal of this project is to discover novel taxa.")
print("In a real-world scenario, many sequences would not have an existing 'label'.")
print("The unclustered sequences (labeled -1 by DBSCAN) are your primary candidates for novel taxa.")
print("These are the sequences that did not belong to any dense cluster of known species.")

# --- 5. Next Steps for Real-World Application ---
print("\n--- Next Steps for Taxonomic Inference ---")
print("To get the actual taxonomic name for a cluster, you would perform a sequence similarity search.")
print("1. Select the representative sequence(s) from a cluster identified above.")
print("2. Use a tool like BLAST  to compare the sequence against a reference database (e.g., NCBI GenBank).")
print("3. The search results will provide the closest known taxonomic matches and percent identity.")
print("4. High similarity (e.g., >97%) suggests the cluster represents that known species.")
print("5. Low similarity or no hits for a cluster of unlabelled sequences indicates a potentially novel taxon.")
print("\n Your pipeline now has all the components for data preparation, unsupervised discovery, and taxonomic inference!")