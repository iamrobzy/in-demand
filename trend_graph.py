import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Path to the folder with date-wise subfolders
base_folder = "./tags"

# Directory to save the plots
output_folder = "./plots"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Initialize data structure to store skill counts
date_skill_counts = {}

# Step 2: Loop through the date folders
for date_folder in sorted(os.listdir(base_folder)):
    folder_path = os.path.join(base_folder, date_folder)
    if os.path.isdir(folder_path):
        # Initialize skill counter for the date
        skill_counter = Counter()
        
        # Loop through all files in the date folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    # Read skills from the file
                    skills = file.read().strip().splitlines()
                    skill_counter.update(skills)
        
        # Save counts for the date
        date_skill_counts[date_folder] = skill_counter

# Step 3: Aggregate the data into a DataFrame
all_dates = sorted(date_skill_counts.keys())
all_skills = set(skill for counts in date_skill_counts.values() for skill in counts)
data = {skill: [date_skill_counts[date].get(skill, 0) for date in all_dates] for skill in all_skills}
df = pd.DataFrame(data, index=all_dates)

print(df)

# Step 4: Identify the top 3 skills
total_counts = df.sum(axis=0)
top_skills = total_counts.nlargest(3).index

# Step 5: Plot and save separate graphs for the top 3 skills
for skill in top_skills:
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df[skill], marker="o", label=skill)
    
    # Add labels and legend
    plt.title(f"Trend of {skill} Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Skill")
    plt.grid()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_folder, f"{skill}_trend.png")
    plt.savefig(plot_path, format="png", dpi=300)
    print(f"Saved plot for {skill} at {plot_path}")
    
    # Show the plot
    plt.show()
