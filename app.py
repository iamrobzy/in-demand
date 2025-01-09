import gradio as gr
from transformers import pipeline
# from embedding_gen import load_skills_from_date, visualize3D
import numpy as np
import pickle

# token_skill_classifier = pipeline(model="jjzha/jobbert_skill_extraction", aggregation_strategy="first")
# token_knowledge_classifier = pipeline(model="jjzha/jobbert_knowledge_extraction")
# token_knowledge_classifier = pipeline(model="Robzy/jobbert_knowledge_extraction")


examples = [
        "High proficiency in Python and AI/ML frameworks, i.e. Pytorch.",
        "Experience with Unreal and/or Unity and/or native IOS/Android 3D development",
        ]


def aggregate_span(results):
    new_results = []
    current_result = results[0]

    for result in results[1:]:
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = result

    new_results.append(current_result)

    return new_results

# def ner(text):


#     output_knowledge = token_knowledge_classifier(text)
#     for result in output_knowledge:
#         if result.get("entity_group"):
#             result["entity"] = "Knowledge"
#             del result["entity_group"]

#     if len(output_knowledge) > 0:
#         output_knowledge = aggregate_span(output_knowledge)

#     return {"text": text, "entities": output_knowledge}

### Visualisation 3D

import os

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
    # os.makedirs(output_folder, exist_ok=True)
    # plot_path = os.path.join(output_folder, f"{date}_3D_clustering.html")
    # fig.write_html(plot_path)
    # print(f"3D clustered plot saved at {plot_path}")
    
    # fig.show()
    return fig



import plotly.express as px
import numpy as np

specific_date = "03-01-2024"  # Example date folder to process
skills = load_skills_from_date('./tags', specific_date)
embeddings = np.load(f"./vectorstore/{specific_date}_embeddings.npy")
with open(f"./vectorstore/{specific_date}_metadata.pkl", "rb") as f:
    metadata =   pickle.load(f)
labels, skills = metadata["labels"], metadata["skills"]
fig = visualize3D(embeddings, labels, skills, n_clusters=5, output_folder="./plots", date=specific_date)
fig.update_layout(
     height=900
)

with gr.Blocks() as demo:
    
    gr.Markdown("# 3D Visualization of Skills in ML Job Postings", elem_id="title")
    # gr.Markdown("Embedding visualisation of sought skills in ML job posting in Stockholm, Sweden on LinkedIn")
    gr.Plot(fig)

    

demo.launch()