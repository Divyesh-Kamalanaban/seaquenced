import numpy as np
import pandas as pd
import random

def generate_asv_data(
    n_known_species=300,
    n_novel_species=200,
    known_asvs_per_species=100,
    novel_asvs_per_species=50,
    asv_length=250,
    random_state=42
):
    """
    Generates a synthetic dataset of ASV sequences.
    Each species has a core pattern with slight mutations.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    base_pairs = ['A', 'C', 'G', 'T']
    asv_sequences = []
    labels = []

    # Generate ASVs for "Known" species with a consistent pattern
    for i in range(n_known_species):
        pattern = ''.join(random.choices(base_pairs, k=asv_length))
        for _ in range(known_asvs_per_species):
            sequence = list(pattern)
            num_mutations = int(asv_length * 0.01) # 1% mutation rate
            mutation_indices = np.random.choice(asv_length, num_mutations, replace=False)
            for idx in mutation_indices:
                original_base = sequence[idx]
                new_base = random.choice([b for b in base_pairs if b != original_base])
                sequence[idx] = new_base
            asv_sequences.append("".join(sequence))
            labels.append(f'Known Species {i}')

    # Generate ASVs for "Novel" species with completely different patterns
    for i in range(n_novel_species):
        pattern = ''.join(random.choices(base_pairs, k=asv_length))
        for _ in range(novel_asvs_per_species):
            sequence = list(pattern)
            num_mutations = int(asv_length * 0.01)
            mutation_indices = np.random.choice(asv_length, num_mutations, replace=False)
            for idx in mutation_indices:
                original_base = sequence[idx]
                new_base = random.choice([b for b in base_pairs if b != original_base])
                sequence[idx] = new_base
            asv_sequences.append("".join(sequence))
            labels.append(f'Novel Species {i}')

    df = pd.DataFrame({
        'asv_sequence': asv_sequences,
        'original_label': labels
    })
    
    return df

if __name__ == '__main__':
    asv_data = generate_asv_data()
    print("Generated ASV dataset (first 5 rows):")
    print(asv_data.head())
    asv_data.to_csv('synthetic_asv_data.csv', index=False)
    print("\nDataset saved to 'synthetic_asv_data.csv'")
