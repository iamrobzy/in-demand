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

This projects aims to monitor in-demand skills for machine learning roles. Skills are extracted with a BERT-based skill extraction model called JobBERT, which is continously fine-tuned on the job postings. The skills are monitored/visualized 1. embedding the extracted skills tokens into vector form, 2. performing dimensionality reduction with UMAP, 3. visualizing the reduced embeddings. 

![Header Image](header.png)

### [Monitoring Platform Link](https://huggingface.co/spaces/jjzha/skill_extraction_demo)

## Architecture & Frameworks


- ** Hugging Face Spaces **
- ** Gradio ** 
- ** GitHub Actions **
- ** Rapid API **
- ** Weight & Biases **
- ** Rapid API **
- ** OpenAI API **


# High-Level Overview

## Model: skills extraction model

## Inference
1. Extracting new job abs from Indeed/LinkedIn
2. Extract skills from job ads via skills extraction model

## Online training
Continual training, extract ground truth via LLM with multi-shot learning with examples. 

## Skill compilation
Save all skills. Make a comprehensive overview by:

1. Embed skills to a vector with an embedding model
2. Perform clustering with KMeans
2. Visualize clustering with dimensionality reduction (UMAP)


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

To monitor the in-demand skills and update our model continously, scheduling is employed. The following scripts are scheduled every Sunday:

1. Job-posting scraping: fetching job descriptions for machine learning from LinkedIn
2. Skills tagging with LLM: we decide to extract the ground truth of skills from the job descriptions by leveraging multi-shot learning and prompt engeneering.
3. Training
4. Embedding and visualizatio -  skills are embedded and visualized with KMeans clustering
