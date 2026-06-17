import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# --- 1. Data Loading and Pre-processing ---

# Load the augmented dataset
try:
    df = pd.read_csv("augmented_deep_sea_edna_dataset.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'augmented_deep_sea_edna_dataset.csv' not found. Please run the augmentation script first.")
    exit()

# Define the one-hot encoding map, including 'N'
BASE_MAP = {"A": [1, 0, 0, 0, 0], 
            "C": [0, 1, 0, 0, 0], 
            "G": [0, 0, 1, 0, 0], 
            "T": [0, 0, 0, 1, 0], 
            "N": [0, 0, 0, 0, 1]}

def one_hot_encode_and_pad(sequences):
    """
    One-hot encodes and pads a list of sequences to a uniform length.
    """
    # Find the maximum sequence length
    max_len = max(len(seq) for seq in sequences)
    print(f"Maximum sequence length detected: {max_len}")
    
    encoded_sequences = []
    for seq in sequences:
        # One-hot encode each nucleotide
        encoded_seq = [BASE_MAP[b] for b in seq]
        
        # Flatten the list of lists into a single array
        flat_encoded_seq = np.array(encoded_seq).flatten()
        
        # Pad the flattened sequence to the max_len * 5 dimensions
        padded_seq = np.pad(flat_encoded_seq, (0, max_len * 5 - len(flat_encoded_seq)), 'constant')
        encoded_sequences.append(padded_seq)
        
    return np.array(encoded_sequences)

# Apply one-hot encoding and padding
X = one_hot_encode_and_pad(df["sequence"])

# Split the data for validation
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# --- 2. Autoencoder Model Building and Training ---

input_dim = X.shape[1]
input_layer = keras.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(128, activation="relu")(input_layer)
encoded = layers.Dropout(0.1)(encoded)
encoded = layers.Dense(64, activation="relu")(encoded)
latent = layers.Dense(2, name="latent")(encoded)

# Decoder
decoded = layers.Dense(64, activation="relu")(latent)
decoded = layers.Dense(128, activation="relu")(decoded)
output_layer = layers.Dense(input_dim, activation="sigmoid")(decoded)

# Autoencoder Model
autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

# *** IMPORTANT FIX: Changed loss function from "mse" to "binary_crossentropy" ***
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print("\n✅ Autoencoder model built and compiled.")
autoencoder.summary()

# Configure early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, # Increased patience to allow more epochs for convergence
    restore_best_weights=True
)

# Train the Autoencoder
print("\n--- Starting Autoencoder Training ---")
history = autoencoder.fit(X_train, X_train,
                          epochs=100, # Increased epochs with Early Stopping
                          batch_size=64, 
                          shuffle=True, 
                          validation_data=(X_val, X_val),
                          callbacks=[early_stopping])

# --- 3. Unsupervised Clustering with DBSCAN ---

# Extract latent space for all original sequences
encoder = keras.Model(inputs=input_layer, outputs=latent)
latent_points = encoder.predict(X)

# Apply DBSCAN clustering on the 2D latent space
# *** SUGGESTED FIX: Tuned eps and min_samples for better results ***
dbscan = DBSCAN(eps=0.15, min_samples=4) # Example of more reasonable parameters
clusters = dbscan.fit_predict(latent_points)

# Add the cluster labels to your latent space DataFrame
df_latent = pd.DataFrame({
    "label": df["label"],
    "x": latent_points[:,0],
    "y": latent_points[:,1],
    "cluster": clusters
})

# Save the latent space with cluster labels
df_latent.to_csv("latent_space_with_clusters.csv", index=False)
print("\n✅ Latent space with clusters saved to 'latent_space_with_clusters.csv'")

# --- 4. Visualization and Interpretation ---

# Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_latent['x'], df_latent['y'], c=df_latent['cluster'], cmap='viridis', s=10)
plt.title("2D Latent Space from Autoencoder with DBSCAN Clusters")
plt.xlabel("Latent Feature 1 (x)")
plt.ylabel("Latent Feature 2 (y)")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.savefig("latent_space_clusters.png")
print("✅ Visualization of clusters saved to 'latent_space_clusters.png'")
plt.show()

# Interpret the results
print("\n--- DBSCAN Cluster Analysis ---")
num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)
print(f"Number of clusters found: {num_clusters}")
print(f"Number of unclustered sequences (potential novel taxa): {num_noise}")

if num_noise > 0:
    print("\n💡 Unclustered sequences (labeled -1) are your potential novel taxa!")
    print("These are sequences that did not fit into a dense cluster of known species.")