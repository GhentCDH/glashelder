import torch
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from umap import UMAP
import logging
import pickle
from hdbscan import HDBSCAN
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import click

MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
# TODO: put some default images in `./images/` as exampels
DEFAULT_IMAGES = "images/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model():
    return CLIPModel.from_pretrained(MODEL)


def get_processor():
    return CLIPProcessor.from_pretrained(MODEL)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else: 
        return 'cpu'


def get_image_embedding(processor, model, device: str, image_path: str):
    image = Image.open(image_path)

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True).to(
            device
        )
        
        return model.get_image_features(**inputs)



@click.command()
def run_clip_with_clustering():
    model = get_model()
    device = get_device()

    # HMM: doesn't processor need to go to device too?
    processor = get_processor()
    model.to(device)

    image_paths = sorted(glob.glob("./selected/*"))

    print("generating embeddings")
    embeddings = torch.cat([get_image_embedding(processor, model, device, path) for path in image_paths])
    print("embeddings generated")

    umap_model = UMAP(n_components=5)
    reduced_embeddings = umap_model.fit_transform(embeddings.cpu().numpy())
    print(reduced_embeddings.shape)
    print(reduced_embeddings)

    # write reduced to pickle file as a checkpoint
    with open("reduced.pkl", "wb") as file:
        pickle.dump(reduced_embeddings, file)

    # with open('reduced.pkl', 'rb') as file:
    # reduced_embeddings = pickle.load(file)

    clusterer = HDBSCAN(min_cluster_size=20)
    print(clusterer.fit(reduced_embeddings))

    # Get cluster labels
    labels = clusterer.labels_

    # Create a scatter plot
    plt.figure(figsize=(12, 8))

    # Use first 2 components for visualization
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]

    # Create scatter plot with different colors for each cluster
    scatter = plt.scatter(x, y, c=labels, cmap="tab20", alpha=0.6, s=50)

    # Add colorbar
    plt.colorbar(scatter, label="Cluster")

    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title(
        f"HDBSCAN Clustering Results\n({len(set(labels)) - (1 if -1 in labels else 0)} clusters, {sum(labels == -1)} noise points)"
    )

    # Add grid
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "clustering_results.png", dpi=300, bbox_inches="tight"
    )  # Save instead of show
    plt.close()  # Close the figure to free memory

    # Print cluster statistics
    unique_labels = set(labels)
    print(
        f"\nNumber of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
    )
    print(f"Number of noise points: {sum(labels == -1)}")
    print("\nCluster sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            print(f"  Noise: {sum(labels == label)} points")
        else:
            print(f"  Cluster {label}: {sum(labels == label)} points")

    # Get cluster labels
    labels = clusterer.labels_

    # Create output directory
    output_dir = Path("clustered_samples")
    output_dir.mkdir(exist_ok=True)

    # Get unique clusters (excluding noise)
    unique_clusters = sorted([label for label in set(labels) if label != -1])

    # For each cluster, copy 2 sample images
    for cluster_id in unique_clusters:
        # Create cluster directory
        cluster_dir = output_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)

        # Get indices of images in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

        # Select up to 2 samples
        sample_indices = cluster_indices[:4]

        # Copy images
        for idx in sample_indices:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = cluster_dir / filename
            shutil.copy(src_path, dst_path)

        print(
            f"Cluster {cluster_id}: {len(cluster_indices)} total images, saved {len(sample_indices)} samples"
        )

    # Optionally handle noise points
    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    if noise_indices:
        noise_dir = output_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        # Save up to 2 noise samples
        for idx in noise_indices[:2]:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = noise_dir / filename
            shutil.copy(src_path, dst_path)

        print(
            f"Noise: {len(noise_indices)} total images, saved {min(2, len(noise_indices))} samples"
        )

    print(f"\nSamples saved to: {output_dir}")
