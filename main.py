from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import os



# Ensure the 'job_descriptions' folder exists
output_folder = "job_descriptions"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the WebDriver (Make sure chromedriver is in your PATH)
driver = webdriver.Chrome()

# Open the Indeed Sweden website
driver.get("https://se.indeed.com/")
time.sleep(2)

# Search for "Machine Learning"
search_box = driver.find_element(By.ID, "text-input-what")
search_box.send_keys("Machine Learning")
search_box.submit()  # Submit the search form
time.sleep(5)

# Extract job information
job_titles = []
job_companies = []
job_locations = []
job_descriptions = []

no_pages = 0

job_cards = driver.find_element(By.ID, "mosaic-provider-jobcards")

# Find the first child element within the parent
first_child = job_cards.find_element(By.XPATH, "./*")  # Selects the first direct child
# Print the text content of the first child
print(first_child)

# Find all child elements within the parent
all_children = first_child.find_elements(By.XPATH, "./*")  # Selects all direct children

# Print the text or other attributes of all child elements
for i in range(1, len(all_children)):
    child = all_children[i]
    print(child.text)  # You can replace `.text` with `.get_attribute('attribute_name')` if needed
    print("*" * 10)
    
    try:
        # Find the div with the class name 'job_seen_beacon' within the parent
        job_div = child.find_element(By.CLASS_NAME, "job_seen_beacon")

        # Click on the div
        job_div.click()
        
        # Locate the element with a class name containing 'jobsearch-Rightpane'
        right_pane_element = driver.find_element(By.XPATH, "//*[contains(@class, 'jobsearch-RightPane')]")

        # Print the text or attributes of the element
        job_description = right_pane_element.text
        print("*******The job description*******")
        print(job_description)  # Or use `.get_attribute('attribute_name')` if needed

        # Save the job description to a text file
        file_name = f"job_description_{i}.txt"
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(job_description)

    except Exception as e:
        print("job beacon doesnt exist")
        
    

    # Scroll slightly down
    driver.execute_script("window.scrollBy(0, 250);")  # Scrolls down by 200 pixels
    time.sleep(1)  # Add a short delay to ensure the page loads correctly
