import pandas as pd
import json
import numpy as np
import random

def run_conceptual_transformer(clustered_data):
    """
    Simulates a pre-trained Transformer model assigning taxonomic names.
    This function maps cluster_ids to names and generates a final report.
    """
    
    # In a real scenario, a Transformer would predict these names.
    # Here, we use a simple mapping based on cluster ID.
    cluster_to_name = {
        0: 'Bathynomus giganteus',
        1: 'Novel Taxa 1',
        2: 'Halobates micans',
        -1: 'Noise'  # DBSCAN marks noise as -1
    }
    
    # Assign names based on cluster ID
    clustered_data['assigned_name'] = clustered_data['cluster_id'].apply(lambda x: cluster_to_name.get(x, f'Unknown Taxa {x}'))

    # Calculate metrics for the report
    total_asvs = len(clustered_data)
    known_asvs = clustered_data[clustered_data['assigned_name'].str.contains('Known')].shape[0]
    novel_asvs = clustered_data[clustered_data['assigned_name'].str.contains('Novel')].shape[0]
    unassigned_asvs = clustered_data[clustered_data['assigned_name'] == 'Noise'].shape[0]

    # Generate a report structure
    report = {
        "total_asvs_processed": total_asvs,
        "known_species_count": known_asvs,
        "novel_taxa_count": novel_asvs,
        "unassigned_reads_count": unassigned_asvs,
        "metrics": {
            "shannon_index": round(np.random.uniform(1.5, 3.5), 2),
            "unassigned_reads_rate": f"{round(unassigned_asvs / total_asvs * 100, 2)}%",
            "avg_confidence": f"{round(np.random.uniform(0.7, 0.99), 2)}",
        },
        "taxonomic_profile": []
    }

    # Generate taxonomic profile from clusters
    for cluster_id, group in clustered_data.groupby('cluster_id'):
        name = group['assigned_name'].iloc[0]
        status = 'Novel' if 'Novel' in name else 'Known' if 'Known' in name else 'Noise'
        
        # Simulate confidence for known taxa
        confidence = round(np.random.uniform(0.8, 0.99), 2) if status == 'Known' else None
        
        report["taxonomic_profile"].append({
            "cluster_id": int(cluster_id),
            "name": name,
            "status": status,
            "read_count": int(len(group)),
            "confidence": confidence,
            "lineage": f"Closest known: {random.choice(['Phylum Chordata', 'Phylum Arthropoda'])}" if status == 'Novel' else "Full classification: ..."
        })

    return report

if __name__ == '__main__':
    # Assume run_clustering.py has been run
    clustered_data = pd.read_json('cluster_results.json')
    
    # Run the conceptual Transformer
    final_report = run_conceptual_transformer(clustered_data)
    
    # Save the report for the frontend
    with open('biodiversity_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)

    print("Final biodiversity report saved to 'biodiversity_report.json'")