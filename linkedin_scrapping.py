import http.client
from config import *
import json
import os
from datetime import datetime


def scrape_jobs():

    conn = http.client.HTTPSConnection("linkedin-job-search-api.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
        'x-rapidapi-host': "linkedin-job-search-api.p.rapidapi.com"
    }

    conn.request("GET", "/active-jb-7d?title_filter=machine%20learning&description_type=text", headers=headers)

    res = conn.getresponse()
    data = res.read()
    jobs_str = data.decode("utf-8")
    jobs = json.loads(jobs_str)

    return jobs

def extract_job_descriptions(jobs):

    # Get the current date in YYYY-MM-DD format and create folder
    current_date = datetime.now().strftime('%d-%m-%Y')
    folder_path = os.path.join("job-postings", current_date)
    os.makedirs(folder_path, exist_ok=True)

    for idx, job in enumerate(jobs, start=1):
        if 'description_text' in job.keys():
            jd = job['description_text']
            print(jd)

            # Save the job description to a text file
            file_path = os.path.join(folder_path, f"{idx}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(jd)
            print("Job {} saved".format(str(idx)))
        else:
            print("Job description not available")

jobs = scrape_jobs()
extract_job_descriptions(jobs)
