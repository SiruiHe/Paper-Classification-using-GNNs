import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def reduce_dimensions(input_file, output_file, n_components=128):
    # Read the CSV file
    print("Loading data...")
    data = pd.read_csv(input_file)
    
    # Standardize the features
    print("Standardizing features...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    print(f"Reducing dimensions from {data.shape[1]} to {n_components}...")
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_scaled)
    
    # Calculate explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total explained variance: {explained_variance:.2f}%")
    
    # Save the reduced data
    print("Saving reduced data...")
    df_reduced = pd.DataFrame(data_reduced)
    df_reduced.to_csv(output_file, index=False)
    
    # Plot explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.savefig('explained_variance.png')
    
    return data_reduced, explained_variance

if __name__ == "__main__":
    dataset_type = "_strict"  # Options: "_strict", "_coarse", "_subseq"
    
    # Construct file paths based on the dataset type
    input_file = f"./construct_dataset/raw/node_features{dataset_type}.csv"
    output_file = f"./construct_dataset/raw/reduced_features{dataset_type}.csv"
    
    reduced_data, variance = reduce_dimensions(input_file, output_file)
    print(f"Dimensionality reduction complete. Output saved to {output_file}")