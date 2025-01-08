import os
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import pickle

# Step 1: Load skills from all files in a specific date folder
def load_skills_from_date(base_folder, date):
    date_folder = os.path.join(base_folder, date)
    all_skills = set()  # To ensure unique skills
    if os.path.exists(date_folder) and os.path.isdir(date_folder):
        for file_name in os.listdir(date_folder):
            file_path = os.path.join(date_folder, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_skills.update(line.strip() for line in f if line.strip())
    return list(all_skills)

# Step 2: Generate embeddings using a pretrained model
def generate_embeddings(skills, model_name="paraphrase-MiniLM-L3-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(skills, convert_to_numpy=True)
    return embeddings

# Step 3: Reduce dimensionality using UMAP
def reduce_dimensions(embeddings, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

# Step 4: Visualize the reduced embeddings (2D)
def visualize_embeddings_2d(reduced_embeddings, skills, output_folder, date):
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, alpha=0.8)
    for i, skill in enumerate(skills):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], skill, fontsize=9, alpha=0.75)
    plt.title(f"UMAP Projection of Skill Embeddings ({date})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{date}_2D_projection.png")
    plt.savefig(plot_path, format="png", dpi=300)
    print(f"2D plot saved at {plot_path}")
    
    plt.show()

# Step 5: Visualize the reduced embeddings (3D)
def visualize_embeddings_3d(reduced_embeddings, skills, output_folder, date):
    fig = px.scatter_3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        text=skills,
        title=f"3D UMAP Projection of Skill Embeddings ({date})"
    )
    
    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{date}_3D_projection.html")
    fig.write_html(plot_path)
    print(f"3D plot saved at {plot_path}")
    
    fig.show()

def visualize3D(reduced_embeddings, labels, skills, n_clusters, output_folder, date):
    
    fig = px.scatter_3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        color=labels,
        text=skills,
        title=f"KMeans Clustering with {n_clusters} Clusters ({date})"
    )
    
    # Save the clustered plot
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{date}_3D_clustering.html")
    fig.write_html(plot_path)
    print(f"3D clustered plot saved at {plot_path}")
    
    # fig.show()
    return fig

# Main execution
base_folder = "./tags"
output_folder = "./plots"
specific_date = "03-01-2024"  # Example date folder to process
# Get today's date in the desired format
# specific_date = datetime.now().strftime("%d-%m-%Y")
n_clusters = 5

    # Main execution
    base_folder = "./tags"
    output_folder = "./plots"
    vector_store = "./vectorstore"
    specific_date = "03-01-2024"  # Example date folder to process
    n_clusters = 5

    # Load skills from the specified date folder
    skills = load_skills_from_date(base_folder, specific_date)
    if not skills:
        print(f"No skills found for the date: {specific_date}")
    else:
        print(f"Loaded {len(skills)} unique skills for the date: {specific_date}")
        
        # Generate embeddings
        embeddings = generate_embeddings(skills)
        
        # Reduce dimensions to 2D and visualize
        # reduced_embeddings_2d = reduce_dimensions(embeddings, n_components=2)
        # visualize_embeddings_2d(reduced_embeddings_2d, skills, output_folder, specific_date)
        
        # Reduce dimensions to 3D, cluster, and visualize
        reduced_embeddings_3d = reduce_dimensions(embeddings, n_components=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced_embeddings_3d)
        visualize3D(reduced_embeddings_3d, labels, skills, n_clusters, output_folder, specific_date)

        # Save the reduced embeddings and metadata
        np.save(os.path.join(vector_store, f"{specific_date}_embeddings.npy"), reduced_embeddings_3d)
        with open(os.path.join(vector_store, f"{specific_date}_metadata.pkl"), 'wb') as f:
            pickle.dump({'labels': labels, 'skills': skills}, f)
        

        # Perform KMeans clustering and visualize in 3D
        # perform_kmeans_and_visualize(reduced_embeddings_3d, skills, n_clusters, output_folder, specific_date)