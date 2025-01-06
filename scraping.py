from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
import time
import os

# Ensure the 'job_descriptions' folder exists
output_folder = "job_descriptions"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the Indeed Sweden website
driver.get("https://se.indeed.com/")
time.sleep(2)

# Search for "Machine Learning"
search_box = driver.find_element(By.ID, "text-input-what")
search_box.send_keys("Machine Learning")
search_box.submit()
time.sleep(5)

# Function to scrape job descriptions on a single page
def scrape_jobs():
    try:
        job_cards = driver.find_element(By.ID, "mosaic-provider-jobcards")
        first_child = job_cards.find_element(By.XPATH, "./*")  # Selects the first direct child
        all_children = first_child.find_elements(By.XPATH, "./*")  # Selects all direct children
        
        for i in range(len(all_children)):
            try:
                child = all_children[i]
                print(f"Processing job {i + 1}:")
                
                # Find the div with the job information
                job_div = child.find_element(By.CLASS_NAME, "job_seen_beacon")
                job_div.click()
                time.sleep(2)  # Allow time for the job description to load
                
                # Extract the job description from the right pane
                right_pane_element = driver.find_element(By.XPATH, "//*[contains(@class, 'jobsearch-RightPane')]")
                job_description = right_pane_element.text
                
                # Save the job description to a text file
                file_name = f"job_description_page_{current_page}_job_{i + 1}.txt"
                file_path = os.path.join(output_folder, file_name)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(job_description)
                
                print(f"Job {i + 1} saved.")
                
            except Exception as e:
                print(f"Error processing job {i + 1}: {e}")
            
            # Scroll slightly down
            driver.execute_script("window.scrollBy(0, 200);")  # Scrolls down by 200 pixels
            time.sleep(1)  # Add a short delay to ensure the page loads correctly
            
    except Exception as e:
        print(f"Error scraping jobs on page {current_page}: {e}")

# Navigate through pages using the pagination block
current_page = 1
while True:
    print(f"Scraping page {current_page}...")
    scrape_jobs()
    
    try:
        # Find the pagination block
        navigation_block = driver.find_element(By.XPATH, "//nav[@aria-label='pagination']")
        first_child = navigation_block.find_element(By.XPATH, "./ul")
        all_children = first_child.find_elements(By.XPATH, "./li")
        
        # Find the "Next" link
        next_link = None
        for child in all_children:
            try:
                link = child.find_element(By.TAG_NAME, "a")
                if "Next" in link.text:
                    next_link = link
                    break
            except Exception as e:
                continue
        
        if next_link:
            next_link.click()
            current_page += 1
            time.sleep(5)  # Allow time for the next page to load
        else:
            print("No more pages to scrape.")
            break
    
    except Exception as e:
        print(f"Error navigating to the next page: {e}")
        break

# Close the browser
driver.quit()
