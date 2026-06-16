import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import json

def run_dbscan_clustering(latent_space_data, eps=1.5, min_samples=2):
    """
    Applies DBSCAN to the latent space data to find clusters.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(latent_space_data[['x', 'y']])
    latent_space_data['cluster_id'] = clusters
    return latent_space_data

def generate_dendrogram(latent_space_data):
    """
    Generates a hierarchical dendrogram to visualize cluster relationships.
    """
    # Filter out noise points (cluster_id = -1) for a cleaner dendrogram
    clustered_points = latent_space_data[latent_space_data['cluster_id'] != -1]
    
    if len(clustered_points) > 1:
        linked = linkage(clustered_points[['x', 'y']], method='ward')
        plt.figure(figsize=(12, 7))
        dendrogram(linked, orientation='top', labels=clustered_points['cluster_id'].astype(str).tolist())
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('ASV Cluster ID')
        plt.ylabel('Distance')
        plt.savefig('dendrogram.png')
    else:
        print("Not enough clustered data points to generate a dendrogram.")
        
    plt.close()

if __name__ == '__main__':
    # Assume autoencoder_simulation.py has been run
    latent_space_data = pd.read_csv('latent_space_data.csv')
    
    # Run DBSCAN
    clustered_data = run_dbscan_clustering(latent_space_data, eps=0.5, min_samples=10)
    print("DBSCAN clusters generated (first 5 rows):")
    print(clustered_data.head())

    # Generate a scatter plot of the clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(clustered_data['x'], clustered_data['y'], c=clustered_data['cluster_id'], cmap='viridis', s=20)
    ax.set_title('DBSCAN Clusters in Latent Space')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True)
    plt.savefig('cluster_plot.png')
    plt.close()
    print("\nScatter plot saved to 'cluster_plot.png'")

    # Generate dendrogram
    generate_dendrogram(clustered_data)
    print("Dendrogram saved to 'dendrogram.png'")

    # Save cluster results for the frontend
    clustered_data[['x', 'y', 'cluster_id']].to_json('cluster_results.json', orient='records')
    print("\nCluster results saved to 'cluster_results.json'")
