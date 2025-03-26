from torch.utils.data import Dataset
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from tqdm import tqdm
import torch
from meshgpt_pytorch import ( 
    MeshAutoencoder,
    MeshTransformer
)


from collections import defaultdict

from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
) 
 
class MeshDataset(Dataset): 
    """
    A PyTorch Dataset to load and process mesh data. 
    The `MeshDataset` provides functions to load mesh data from a file, embed text information, generate face edges, and generate codes.

    Attributes:
        data (list): A list of mesh data entries. Each entry is a dictionary containing the following keys:
            vertices (torch.Tensor): A tensor of vertices with shape (num_vertices, 3).
            faces (torch.Tensor): A tensor of faces with shape (num_faces, 3).
            text (str): A string containing the associated text information for the mesh.
            text_embeds (torch.Tensor): A tensor of text embeddings for the mesh.
            face_edges (torch.Tensor): A tensor of face edges with shape (num_faces, num_edges).
            codes (torch.Tensor): A tensor of codes generated from the mesh data.

    Example usage:

    ``` 
    data = [
        {'vertices': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), 'faces': torch.tensor([[0, 1, 2]], dtype=torch.long), 'text': 'table'},
        {'vertices': torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), 'faces': torch.tensor([[1, 2, 0]], dtype=torch.long), "text": "chair"},
    ]

    # Create a MeshDataset instance
    mesh_dataset = MeshDataset(data)

    # Save the MeshDataset to disk
    mesh_dataset.save('mesh_dataset.npz')

    # Load the MeshDataset from disk
    loaded_mesh_dataset = MeshDataset.load('mesh_dataset.npz')
    
    # Generate face edges so it doesn't need to be done every time during training
    dataset.generate_face_edges()
    ```
    """
    def __init__(self, data): 
        self.data = data 
        print(f"[MeshDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        data = self.data[idx] 
        return data  
    
    def save(self, path):  
        np.savez_compressed(path, self.data, allow_pickle=True) 
        print(f"[MeshDataset] Saved {len(self.data)} entries at {path}")
        
    @classmethod
    def load(cls, path):   
        loaded_data = np.load(path, allow_pickle=True)  
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)  
        print(f"[MeshDataset] Loaded {len(data)} entries")
        return cls(data) 
    
    def sort_dataset_keys(self):
        desired_order = ['vertices', 'faces', 'face_edges', 'texts','text_embeds','codes'] 
        self.data = [
            {key: d[key] for key in desired_order if key in d} for d in self.data
        ]
       
    def generate_face_edges(self, batch_size = 5):   
        data_to_process = [item for item in self.data if 'faces_edges' not in item]
        
        total_batches = (len(data_to_process) + batch_size - 1) // batch_size 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for i in tqdm(range(0, len(data_to_process), batch_size), total=total_batches):
            batch_data = data_to_process[i:i+batch_size]  
            
            if not batch_data:  
                continue
            
            padded_batch_faces = pad_sequence(
                [item['faces'] for item in batch_data], 
                batch_first=True, 
                padding_value=-1
            ).to(device)
            
            batched_faces_edges = derive_face_edges_from_faces(padded_batch_faces, pad_id=-1)

            mask = (batched_faces_edges != -1).all(dim=-1) 
            for item_idx, (item_edges, item_mask) in enumerate(zip(batched_faces_edges, mask)):            
                item_edges_masked = item_edges[item_mask]
                item = batch_data[item_idx]
                item['face_edges'] = item_edges_masked 

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated face_edges for {len(data_to_process)} entries")

    def generate_codes(self, autoencoder : MeshAutoencoder, batch_size = 25): 
        total_batches = (len(self.data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(self.data), batch_size), total=total_batches):
            batch_data = self.data[i:i+batch_size] 
            
            padded_batch_vertices = pad_sequence([item['vertices'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).to(autoencoder.device)
            padded_batch_faces = pad_sequence([item['faces'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).to(autoencoder.device)
            padded_batch_face_edges = pad_sequence([item['face_edges'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).to(autoencoder.device)
            
            batch_codes = autoencoder.tokenize(
                vertices=padded_batch_vertices,
                faces=padded_batch_faces,
                face_edges=padded_batch_face_edges
            )
            

            mask = (batch_codes != autoencoder.pad_id).all(dim=-1) 
            for item_idx, (item_codes, item_mask) in enumerate(zip(batch_codes, mask)):            
                item_codes_masked = item_codes[item_mask]
                item = batch_data[item_idx]
                item['codes'] = item_codes_masked 
                
        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated codes for {len(self.data)} entries")
    
    def embed_texts(self, transformer : MeshTransformer, batch_size = 50): 
        unique_texts = list(set(item['texts'] for item in self.data)) 
        text_embedding_dict = {}
        for i in tqdm(range(0,len(unique_texts), batch_size)):
            batch_texts = unique_texts[i:i+batch_size]
            text_embeddings = transformer.embed_texts(batch_texts)
            mask = (text_embeddings != transformer.conditioner.text_embed_pad_value).all(dim=-1)  
            
            for idx, text in enumerate(batch_texts): 
                masked_embedding = text_embeddings[idx][mask[idx]]
                text_embedding_dict[text] = masked_embedding
                
        for item in self.data: 
            if 'texts' in item:  
                item['text_embeds'] = text_embedding_dict.get(item['texts'], None)
                del item['texts'] 
                
        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated {len(text_embedding_dict)} text_embeddings") 

    def faces_statistics(self):
        """
        Calculates the average and standard deviation of the number of faces per model in the dataset.

        Returns:
            tuple: (average_faces, std_faces) where:
                - average_faces (float): The average number of faces per model.
                - std_faces (float): The standard deviation of the number of faces per model.
        """
        if not self.data:
            print("[MeshDataset] Dataset is empty. Cannot compute face statistics.")
            return 0.0, 0.0

        face_counts = [item['faces'].shape[0] for item in self.data if 'faces' in item]

        if not face_counts:
            print("[MeshDataset] No models with faces found. Cannot compute statistics.")
            return 0.0, 0.0

        median_faces = np.median(face_counts)
        mode_faces = stats.mode(face_counts)


        average_faces = np.mean(face_counts)
        std_faces = np.std(face_counts)

        print(f"[MeshDataset] Average faces per model: {average_faces:.2f}")
        print(f"[MeshDataset] Standard deviation of faces per model: {std_faces:.2f}")
        print(f"[MeshDataset] Median faces: {median_faces:.2f}")
        print(f"[MeshDataset] Mode faces: {mode_faces:.2f}")

        return average_faces, std_faces

    def draw_diagram(face_counts, plot_kde=True, save_path="renders/face_counts_distribution.png", density=False):
        """
        Draws a histogram of the distribution of face counts with an optional kernel density estimate (KDE)
        overlay, and saves the figure to a file.

        Args:
            face_counts (list or array): A list or array containing the face counts, or dictionaries 
                                        where each dictionary has a 'faces' key.
            plot_kde (bool): If True, overlays a KDE curve on the histogram. (Typically used when density=True.)
            save_path (str): File path to save the generated figure.
            density (bool): If True, the histogram will be normalized to represent probability density;
                            if False, it will show raw counts.
        """
        # If the input is a list of dictionaries, extract the face counts.
        if face_counts and isinstance(face_counts[0], dict):
            extracted_face_counts = []
            for item in face_counts:
                if 'faces' in item and hasattr(item['faces'], 'shape'):
                    extracted_face_counts.append(item['faces'].shape[0])
                else:
                    print("Skipping item without proper 'faces' key or shape attribute.")
            face_counts = extracted_face_counts

        # Convert to a NumPy array of float values.
        face_counts = np.asarray(face_counts, dtype=np.float64)

        plt.figure(figsize=(8, 6))
        # Plot histogram; if density=True, the y-axis will show probability density.
        n, bins, patches = plt.hist(face_counts, bins='auto', density=density,
                                    alpha=0.6, color='g', edgecolor='black')
        plt.title('Distribution of Face Counts')
        plt.xlabel('Number of Faces')
        plt.ylabel('Density' if density else 'Count')

        # Only plot the KDE overlay if using density normalization.
        if plot_kde and density:
            density_est = gaussian_kde(face_counts)
            xs = np.linspace(face_counts.min(), face_counts.max(), 200)
            plt.plot(xs, density_est(xs), 'k--', label='KDE')
            plt.legend()

        plt.savefig(save_path)
        print(f"Saved histogram at: {save_path}")
        plt.close()
