import os
import pandas as pd
import json
# Ensure that generate_asv_data.py exists in the same directory or in your PYTHONPATH
from generate_asv_data import generate_asv_data
from autoencoder_simulation import simulate_autoencoder_output
from run_clustering import run_dbscan_clustering, generate_dendrogram
from conceptual_transformer import run_conceptual_transformer

def create_output_directory(dir_name='data_output'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

if __name__ == '__main__':
    output_dir = create_output_directory()
    os.chdir(output_dir)

    print("--- Starting Argonauts Prototype Pipeline ---")
    
    # Step 1: Generate Synthetic ASV Data
    print("1. Generating synthetic ASV data...")
    asv_data = generate_asv_data()
    asv_data.to_csv('synthetic_asv_data.csv', index=False)
    
    # Step 2: Simulate Autoencoder Output
    print("2. Simulating Autoencoder output...")
    latent_space_data = simulate_autoencoder_output(asv_data)
    latent_space_data.to_csv('latent_space_data.csv', index=False)

    # Step 3: Run Clustering (DBSCAN & Hierarchical)
    print("3. Running DBSCAN and generating visualizations...")
    clustered_data = run_dbscan_clustering(latent_space_data, eps=0.5, min_samples=10)
    clustered_data.to_json('cluster_results.json', orient='records')
    generate_dendrogram(clustered_data)
    
    # Step 4: Run Conceptual Transformer and Generate Report
    print("4. Running conceptual Transformer and compiling report...")
    final_report = run_conceptual_transformer(clustered_data)
    with open('biodiversity_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)

    print("\n--- Pipeline Complete! ---")
    print("All output files (plots, JSON reports) are in the 'data_output' directory.")