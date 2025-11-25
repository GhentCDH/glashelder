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
from flask import Flask, render_template, request, send_from_directory

from dataloader import get_dataloader

MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
# TODO: put some default images in `./images/` as exampels
DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
# actual dataset location
IMAGE_DIR = "/mnt/miepeete/shares/glashelder/assets"
EMBEDDINGS_PATH = "embeddings.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model():
    return CLIPModel.from_pretrained(MODEL)


def get_processor():
    return CLIPProcessor.from_pretrained(MODEL)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else: 
        return 'cpu'


def reduce_dimensions(embeddings, dimensions: int = 5):
    umap_model = UMAP(n_components=5)
    return umap_model.fit_transform(embeddings)


def cluster(latent, min_cluster_size: int = 20):
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(latent)

    return clusterer


def get_image_embedding(processor, model, device: str, image_path: str):
    image = Image.open(image_path)

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True).to(
            device
        )
        
        return model.get_image_features(**inputs)


def get_text_embedding(processor, model, device: str, text: str):
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        return model.get_text_features(**inputs)


def create_scatter_plot(latent, labels, save_path: str = "clustering_results.png"):
    # Create a scatter plot
    plt.figure(figsize=(12, 8))

    # Use first 2 components for visualization
    x = latent[:, 0]
    y = latent[:, 1]

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
        save_path, dpi=300, bbox_inches="tight"
    )  # Save instead of show
    plt.close()  # Close the figure to free memory


def make_embeddings(model, processor, image_paths, batch_size, device):
    dataloader = get_dataloader(image_paths, processor, batch_size=batch_size)
    all_embeddings = []
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i + 1}/{len(dataloader)}")
        
        # Move batch to GPU
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        all_embeddings.append(embeddings.cpu())
    
    embeddings = torch.cat(all_embeddings)

    return embeddings


def perform_search(top_k, text_embedding, image_embeddings, image_paths):
    """Shared search logic"""
    # Normalize embeddings
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarities = (text_embedding @ image_embeddings_norm.T).squeeze(0)
    
    # Get top-k indices
    top_k = min(top_k, len(image_paths))
    top_values, top_indices = torch.topk(similarities, top_k)
    
    return top_indices, top_values


def find_similar(query_index, top_k, image_embeddings, image_paths, exclude_self=True):
    """Find images similar to a given image by index in the embedding space"""
    query_embedding = image_embeddings[query_index].unsqueeze(0)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    similarities = (query_embedding @ image_embeddings_norm.T).squeeze(0)
    
    if exclude_self:
        similarities[query_index] = -float('inf')
        actual_top_k = min(top_k, len(image_paths) - 1)
    else:
        actual_top_k = min(top_k, len(image_paths))
    
    top_values, top_indices = torch.topk(similarities, actual_top_k)
    return top_indices, top_values


@click.group()
def cli():
    pass


@click.command()
@click.option('--image-dir', default=IMAGE_DIR, help='Directory containing images to search')
def run_clip_with_clustering(image_dir):
    model = get_model()
    device = get_device()

    # HMM: doesn't processor need to go to device too?
    processor = get_processor()
    model.to(device)

    image_paths = sorted(glob.glob(f"{image_dir}/*"))

    logger.info("generating embeddings")
    embeddings = torch.cat([get_image_embedding(processor, model, device, path) for path in image_paths])
    logger.info("embeddings generated")

    reduced_embeddings = reduce_dimensions(embeddings.cpu().numpy())
    logger.info(reduced_embeddings.shape)
    logger.info(reduced_embeddings)

    # write reduced to pickle file as a checkpoint
    with open("reduced.pkl", "wb") as file:
        pickle.dump(reduced_embeddings, file)

    # with open('reduced.pkl', 'rb') as file:
    # reduced_embeddings = pickle.load(file)

    clusterer = cluster(reduced_embeddings)

    # Get cluster labels
    labels = clusterer.labels_

    create_scatter_plot(reduced_embeddings, labels)

    # Print cluster statistics
    unique_labels = set(labels)
    logger.info(
        f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
    )
    logger.info(f"Number of noise points: {sum(labels == -1)}")
    logger.info("Cluster sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            logger.info(f"  Noise: {sum(labels == label)} points")
        else:
            logger.info(f"  Cluster {label}: {sum(labels == label)} points")

    # Create output directory
    output_dir = Path(CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR)
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


@cli.command()
@click.argument('prompt')
@click.option('--top-k', default=10, help='Number of top matching images to return')
@click.option('--image-dir', default=IMAGE_DIR, help='Directory containing images to search')
@click.option('--output-dir', default='search_results', help='Directory to save matching images')
@click.option('--sample-size', default=10000, help='The amount of images in the dataset to consider')
@click.option('--embeddings', default=None, help='The path to a embeddings pickle file')
@click.option('--batch-size', default=64, help='Batch size for embedding generation')
def search(prompt, top_k, image_dir, output_dir, sample_size, embeddings, batch_size):
    """Search for images matching a text prompt using CLIP similarity"""
    logger.info(f"Searching for: '{prompt}'")
    logger.info(f"Looking for top {top_k} matches")
    
    model = get_model()
    device = get_device()
    processor = get_processor()
    model.to(device)

    # Get text embedding for the prompt
    logger.info("Generating text embedding...")
    text_embedding = get_text_embedding(processor, model, device, prompt)

    # Get image embeddings
    if embeddings is not None:
        with open(embeddings, 'rb') as file:
            data = pickle.load(file)
            image_paths = data["images"]
            image_embeddings = data["embeddings"].to(device)
    else:
        # Get all image paths
        image_paths = list(glob.glob(f"{image_dir}/*"))[:sample_size]
        image_embeddings = make_embeddings(model, processor, image_paths, batch_size, device)
        logger.info("Generating image embeddings...")

    top_indices, top_scores = perform_search(top_k, text_embedding, image_embeddings, image_paths)

    # Print results
    logger.info(f"\nTop {top_k} matches for '{prompt}':")
    for i, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()), 1):
        logger.info(f"{i}. {Path(image_paths[idx]).name} (similarity: {score:.4f})")

    # Create output directory and copy images
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clear previous results
    for old_file in output_path.glob("*"):
        old_file.unlink()

    for i, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()), 1):
        src_path = image_paths[idx]
        filename = Path(src_path).name
        # Add rank and score to filename
        dst_filename = f"{i:02d}_score{score:.3f}_{filename}"
        dst_path = output_path / dst_filename
        shutil.copy(src_path, dst_path)

    logger.info(f"\nTop {top_k} images saved to: {output_path}")


@cli.command()
@click.option('--image-dir', default=IMAGE_DIR, help='Directory containing images to search')
@click.option('--save-path', default=EMBEDDINGS_PATH, help='Embeddings pickle save path')
@click.option('--count', default=10000, help='Amount of images to generate embeddings for')
@click.option('--batch-size', default=64, help='Batch size for embedding generation')
def generate_embeddings(image_dir, save_path, count, batch_size):
    model = get_model()
    device = get_device()
    processor = get_processor()
    model.to(device)
    
    image_paths = sorted(glob.glob(f"{image_dir}/*"))[:count]
    logger.info(f"Generating embeddings for {len(image_paths)} images with batch size {batch_size}")

    embeddings = make_embeddings(model, processor, image_paths, batch_size, device)
    
    with open(save_path, "wb") as file:
        pickle.dump({"images": image_paths, "embeddings": embeddings}, file)


@cli.command()
@click.option('--embeddings', default=EMBEDDINGS_PATH, help='Path to embeddings pickle file')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=5000, help='Port to bind to')
def serve(embeddings, host, port):
    """Start a web server for interactive image search"""
    app = Flask(__name__)
    
    # Load model and embeddings once
    logger.info("Loading model and processor...")
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model = get_model()
    processor = get_processor()
    model.to(device)
    model.eval()
    
    logger.info(f"Loading embeddings from {embeddings}...")
    with open(embeddings, 'rb') as f:
        data = pickle.load(f)
        image_paths = data["images"]
        image_embeddings = data["embeddings"].to(device)
    
    logger.info(f"Loaded {len(image_paths)} images")
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/search', methods=['POST'])
    def search_endpoint():
        prompt = request.form.get('prompt', '').strip()
        top_k = int(request.form.get('top_k', 12))
        
        if not prompt:
            return render_template('results.html', error="Please enter a search prompt", results=[], query_type='text', query_value='')
        
        try:
            text_embedding = get_text_embedding(processor, model, device, prompt)
            top_indices, top_values = perform_search(top_k, text_embedding, image_embeddings, image_paths)
            
            results = []
            for rank, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy()), 1):
                results.append({
                    'rank': rank,
                    'index': int(idx),
                    'score': float(score),
                    'filename': Path(image_paths[idx]).name
                })
            
            return render_template('results.html', results=results, error=None, query_type='text', query_value=prompt)
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return render_template('results.html', error=f"Search failed: {str(e)}", results=[], query_type='text', query_value='')
    
    @app.route('/similar/<int:index>')
    def similar_endpoint(index):
        """Find images similar to the image at the given index"""
        top_k = int(request.args.get('top_k', 12))
        
        if index < 0 or index >= len(image_paths):
            return render_template('results.html', error="Image not found", results=[], query_type='similar', query_value=index)
        
        try:
            top_indices, top_values = find_similar(index, top_k, image_embeddings, image_paths)
            
            results = []
            for rank, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy()), 1):
                results.append({
                    'rank': rank,
                    'index': int(idx),
                    'score': float(score),
                    'filename': Path(image_paths[idx]).name
                })
            
            query_filename = Path(image_paths[index]).name
            return render_template('results.html', results=results, error=None, query_type='similar', query_value=index, query_filename=query_filename)
        
        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return render_template('results.html', error=f"Search failed: {str(e)}", results=[], query_type='similar', query_value=index)
    
    @app.route('/image/<int:index>')
    def serve_image(index):
        """Serve image by index"""
        if index < 0 or index >= len(image_paths):
            return "Image not found", 404
        
        image_path = Path(image_paths[index])
        return send_from_directory(image_path.parent, image_path.name)
    
    logger.info(f"Starting Flask server on {host}:{port}...")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    cli()
