import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def simulate_autoencoder_output(asv_data, n_components=2, random_state=42):
    """
    Simulates an autoencoder's output by converting ASV strings into 2D coordinates.
    This uses a TF-IDF vectorizer and t-SNE for dimensionality reduction,
    which conceptually mimics the function of an autoencoder.
    """
    np.random.seed(random_state)

    # Convert ASV strings to numerical features using TF-IDF Vectorizer
    # This is a good way to represent strings as numbers for ML models
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    asv_features = vectorizer.fit_transform(asv_data['asv_sequence'])

    # Use t-SNE to reduce the high-dimensional data to 2D
    # This is a powerful technique for visualizing high-dimensional data
    tsne = TSNE(n_components=n_components, random_state=random_state, init="random")
    latent_space_coords = tsne.fit_transform(asv_features)

    # Combine with original labels for validation (DBSCAN will ignore this)
    latent_space_df = pd.DataFrame(latent_space_coords, columns=['x', 'y'])
    latent_space_df['original_label'] = asv_data['original_label']

    return latent_space_df

if __name__ == '__main__':
    # Assume generate_asv_data.py has been run and created 'synthetic_asv_data.csv'
    asv_data = pd.read_csv('synthetic_asv_data.csv')
    latent_space_data = simulate_autoencoder_output(asv_data)

    print("Simulated Autoencoder output (first 5 rows):")
    print(latent_space_data.head())
    
    latent_space_data.to_csv('latent_space_data.csv', index=False)
    print("\nLatent space data saved to 'latent_space_data.csv'")
