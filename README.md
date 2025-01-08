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


# Compilation of in-demand tech skills

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
2. Perform clustering with HDBSCAN
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





1. Scraping
2. Tagging of JP
    - tag date
3. Training
4. Visualisation