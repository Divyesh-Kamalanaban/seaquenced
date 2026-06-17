import pandas as pd
import random
import csv

def generate_synthetic_sequences(sequence, num_sequences=20, mutation_rate=0.01):
    """
    Generates synthetic sequences with random mutations.

    Args:
        sequence (str): The original DNA sequence.
        num_sequences (int): The number of new sequences to generate.
        mutation_rate (float): The probability of a mutation at each nucleotide.

    Returns:
        list: A list of newly generated synthetic sequences.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    synthetic_sequences = []

    for _ in range(num_sequences):
        mutated_sequence = list(sequence)
        for i in range(len(mutated_sequence)):
            if random.random() < mutation_rate:
                original_nuc = mutated_sequence[i]
                
                # Choose a new nucleotide that is different from the original
                new_nuc = random.choice([n for n in nucleotides if n != original_nuc])
                mutated_sequence[i] = new_nuc

        synthetic_sequences.append("".join(mutated_sequence))
    
    return synthetic_sequences

def main():
    """
    Main function to read the CSV, augment the data, and save the new CSV.
    """
    input_file = "deep_sea_edna_dataset.csv"
    output_file = "augmented_deep_sea_edna_dataset.csv"
    
    # --- Configuration ---
    # Number of new synthetic sequences to generate for EACH original sequence.
    # Adjust this value to reach your target of 100-200 sequences per species.
    # If your average file has ~15 sequences, generating 10 new ones per sequence
    # would give you ~150 sequences per species.
    sequences_to_generate_per_original = 10
    
    # Mutation rate (e.g., 0.01 means 1% of nucleotides will be mutated).
    mutation_rate = 0.01 
    
    # List to hold all original and new augmented sequences
    augmented_data = []

    try:
        # Read the original CSV dataset
        df = pd.read_csv(input_file)
        
        # Iterate over each row (each original sequence) in the DataFrame
        for index, row in df.iterrows():
            label = row['label']
            sequence = row['sequence']
            
            # Add the original sequence to the new dataset
            augmented_data.append({'label': label, 'sequence': sequence})
            
            # Generate new synthetic sequences
            new_sequences = generate_synthetic_sequences(sequence, 
                                                         num_sequences=sequences_to_generate_per_original, 
                                                         mutation_rate=mutation_rate)
            
            # Add the newly generated sequences to the dataset with the same label
            for new_seq in new_sequences:
                augmented_data.append({'label': label, 'sequence': new_seq})
                
        # Save the combined dataset to a new CSV file
        print(f"Writing {len(augmented_data)} total sequences to {output_file}...")
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['label', 'sequence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in augmented_data:
                writer.writerow(row)
        
        print("Augmentation complete! 🎉 The new augmented dataset is ready.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()