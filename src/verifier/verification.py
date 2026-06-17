import pandas as pd
import numpy as np

# Load the latent space data with cluster labels
try:
    df_latent = pd.read_csv("latent_space_with_clusters.csv")
    print("✅ Latent space data loaded successfully.")
except FileNotFoundError:
    print("Error: 'latent_space_with_clusters.csv' not found. Please run the full script first.")
    exit()

print("\n--- Verifying DBSCAN Clusters ---")

# Count the number of unique species labels in the original dataset
unique_species = df_latent['label'].nunique()
print(f"Total number of unique species labels in the dataset: {unique_species}\n")

# Group the data by cluster ID
cluster_groups = df_latent.groupby('cluster')

for cluster_id, group in cluster_groups:
    if cluster_id == -1:
        # This is the noise cluster, our potential novel taxa
        print(f"Cluster ID: {cluster_id} (Noise / Potential Novel Taxa)")
    else:
        print(f"Cluster ID: {cluster_id}")
    
    # Check the unique species labels within this cluster
    species_in_cluster = group['label'].unique()
    num_species_in_cluster = len(species_in_cluster)
    
    # Check the number of sequences in this cluster
    num_sequences_in_cluster = len(group)
    
    print(f"  - Number of sequences: {num_sequences_in_cluster}")
    
    if num_species_in_cluster == 1:
        print(f"  - This cluster is homogeneous (contains one species): {species_in_cluster[0]}")
    elif num_species_in_cluster > 1:
        print(f"  - This cluster is heterogeneous (contains {num_species_in_cluster} species): {species_in_cluster}")
    else:
        print(f"  - No species found in this cluster.")
    
    print("-" * 50)

print("\n✅ Verification complete.")