---
title: Tech skill demand
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---


# In-demand Skill Monitoring for Machine Learning Industry

## About

This projects strives to monitor in-demand skills for machine learning roles based in Stockholm, Sweden. 

# Project outline

## Model: skills extraction model

[Model: skills extraction model from HuggingFace](https://huggingface.co/spaces/jjzha/skill_extraction_demo)

## Inference
1. Extracting new job abs from Indeed/LinkedIn
2. Extract skills from job ads via skills extraction model

## Online training
Extract ground truth via LLM and few-shot learning. 

## Skill compilation
Save all skills. Make a comprehensive overview by:

1. Embed skills to a vector with an embedding model
2. Perform clustering with KMeans
2. Visualize clustering with dimensionality reduction (UMAP)
    
Inspiration: [link](https://dylancastillo.co/posts/clustering-documents-with-openai-langchain-hdbscan.html)


## Project requirements:

You should define your own project by writing at most one page description of the project. The proposed project should be approved by the examiner. The project proposal should cover the following headings:

### Problem description: what are the data sources and the prediction problem that you will be building a ML System for?
### Tools: what tools you are going to use? In the course we mainly used Decision Trees and PyTorch/Tensorflow, but you are free to explore new tools and technologies.
### Data: what data will you use and how are you going to collect it?
### Methodology and algorithm: what method(s) or algorithm(s) are you proposing?
### What to deliver
You should deliver your project as a stand alone serverless ML system. You should submit a URL for your service, a zip file containing your code, and a short report (two to three pages) about what you have done, the dataset, your method, your results, and how to run the code. I encourage you to have the README.md for your project in your Github report as the report for your project.


# Job Scraping

This component scrapes job descriptions from the LinkedIn Job Search API for Machine Learning, and saves them in text files for further analysis.

## Workflow

1. **API Configuration**:
   - The script uses the `linkedin-job-search-api.p.rapidapi.com` endpoint to fetch job data.
   - API access is authenticated using a RapidAPI key stored as an environment variable `RAPID_API_KEY`.

2. **Data Retrieval**:
   - The script fetches jobs matching the keyword `machine learning`.
   - It retrieves job details including the description, which is saved for further analysis.

3. **Job Description Extraction**:
   - Each job description is saved in a `.txt` file under the `job-postings/<date>` folder.
   
# Skill Embeddings and Visualization

We generate embeddings for technical skills listed in .txt files and visualizes their relationships using dimensionality reduction and clustering techniques. The visualizations are created for both 2D and 3D embeddings, and clustering is performed using KMeans to identify groups of similar skills.

## Workflow

### 1. Input Data
- Skills are loaded from `.txt` files located in date-based subfolders under the `./tags` directory.
- Each subfolder corresponds to a specific date (e.g., `03-01-2024`).

### 2. Embedding Generation
- The script uses the `SentenceTransformer` model (`paraphrase-MiniLM-L3-v2`) to generate high-dimensional embeddings for the unique skills.

### 3. Dimensionality Reduction
- UMAP (Uniform Manifold Approximation and Projection) is used to reduce the embeddings to:
  - **2D**: For creating simple scatter plots.
  - **3D**: For interactive visualizations.

### 4. Clustering
- KMeans clustering is applied to the 3D embeddings to group similar skills into clusters.
- The number of clusters can be specified in the script.

### 5. Visualization and Outputs
- **2D Projection**: Saved as PNG images in the `./plots` folder.
- **3D Projection**: Saved as interactive HTML files in the `./plots` folder.
- **3D Clustering Visualization**: Saved as HTML files, showing clusters with different colors.

# Scheduling

- scrapping: We run scrapping weekly to fetch job descriptions for machine learning from LinkedIn
- LLM tagging:
- Training:
- Embedding and visualization: On weekly basis, we also use the skills extracted to create their embeddings and visualize them using KMeans clustering

