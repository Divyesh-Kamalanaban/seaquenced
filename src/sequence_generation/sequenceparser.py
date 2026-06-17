import os
import re
import csv

def parse_fasta(file_path):
    """
    Parses a FASTA-formatted text file and extracts sequences.
    
    Args:
        file_path (str): The path to the input FASTA file.
    
    Returns:
        list: A list of tuples, where each tuple contains
              (species_label, sequence_string).
    """
    
    # Extract the species name from the filename for labeling
    file_name = os.path.basename(file_path)
    # This logic assumes the species name is everything before "COI.txt"
    # and replaces underscores with spaces.
    species_label = file_name.replace("COI.txt", "").replace("_", " ").strip()
    
    labeled_sequences = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        entries = content.split('>')
        
        for entry in entries[1:]:
            lines = entry.split('\n')
            header = lines[0]
            
            # Find the gene name from the header line
            match = re.search(r'\[gene=(.+?)\]', header)
            if match:
                gene_name = match.group(1)
            else:
                gene_name = "unknown"
            
            # Join the sequence lines, remove whitespace and convert to uppercase
            sequence = "".join(lines[1:]).replace(" ", "").upper()
            
            if sequence:
                # We'll use the species label extracted from the filename
                # and a combination of the species and gene for more specific labeling
                # e.g., "Riftia pachyptila_COI"
                combined_label = f"{species_label}_{gene_name}"
                labeled_sequences.append({'label': combined_label, 'sequence': sequence})
                
    return labeled_sequences

def main():
    """
    Main function to process all files and write a consolidated CSV file.
    """
    # A list of the file paths for your species
    file_list = [
        "RiftiapachyptilaCOI.txt",
        "IdiacanthusantrostomusCOI.txt",
        "VampyroteuthisinfernalisCOI.txt",
        "BenthosemaglacialeCOI.txt",
        "PhysetermacrocephalusCOI.txt",
        "AlviniconchahessleriCOI.txt",
        # Add any other file names here
    ]
    
    # List to hold all labeled sequences from all files
    all_sequences = []
    
    for file_path in file_list:
        if os.path.exists(file_path):
            print(f"Parsing file: {file_path}")
            labeled_sequences = parse_fasta(file_path)
            all_sequences.extend(labeled_sequences)
        else:
            print(f"File not found: {file_path}")
            
    if all_sequences:
        output_file = "deep_sea_edna_dataset.csv"
        print(f"\nWriting {len(all_sequences)} sequences to {output_file}...")
        
        # Write the data to a CSV file
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['label', 'sequence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in all_sequences:
                writer.writerow(row)
        
        print("CSV file created successfully! 🎉")
    else:
        print("\nNo sequences were found to write to a CSV file.")

if __name__ == "__main__":
    main()