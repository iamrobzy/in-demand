# Generating sample folder structure and files with multiple skills per file

import os

# Base folder for the structure
base_folder = "tags"

# Sample data: dates and skills for each date
sample_dates = ["03-01-2024", "04-01-2024", "05-01-2024"]
sample_skills = {
    "03-01-2024": [
        ["Python", "Machine Learning", "Data Analysis"],
        ["Python", "Deep Learning"],
        ["Data Science", "AI"]
    ],
    "04-01-2024": [
        ["Python", "AI", "Data Analysis"],
        ["Deep Learning", "Machine Learning"],
        ["AI", "Data Engineering"]
    ],
    "05-01-2024": [
        ["AI", "Machine Learning", "Python"],
        ["Data Science", "Deep Learning"],
        ["Python", "AI", "Cloud Computing"]
    ]
}

# Create the folder structure and files
for date in sample_dates:
    date_folder = os.path.join(base_folder, date)
    os.makedirs(date_folder, exist_ok=True)
    
    for i, skills in enumerate(sample_skills[date], start=1):
        file_path = os.path.join(date_folder, f"{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skills))

print(f"Sample files with multiple skills per file have been generated in the '{base_folder}' folder.")
