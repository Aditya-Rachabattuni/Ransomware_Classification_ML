Ransomware Classification and Detection using Machine Learning
This project is a machine learning-powered application designed to classify executable files as safe or ransomware-infected. The system utilizes classification algorithms such as Decision Tree and Random Forest to detect potential ransomware threats by analyzing key file attributes.

Project Overview
Purpose: To enhance cybersecurity by identifying and classifying ransomware using machine learning.
Tech Stack: Python, Django, SQLite, HTML, CSS, JavaScript.
ML Algorithms: Decision Tree, Random Forest.
Features
Real-Time Classification: Upload an executable file, and the application analyzes and classifies it in real-time.
User-Friendly Interface: Built with Django, HTML, and CSS for easy interaction and clean data display.
Data Storage: Uses SQLite to store analysis results, making it easy to track and retrieve classified files.
Model Accuracy: Achieved high accuracy in classifying files based on their attributes.
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/Aditya-Rachabattuni/Ransomware_Classification_ML.git
cd Ransomware_Classification_ML
Install Dependencies Ensure you have Python installed, then install required packages:

bash
Copy code
pip install -r requirements.txt
Set Up Database Migrate the SQLite database:

bash
Copy code
python manage.py migrate
Run the Application Start the Django development server:

bash
Copy code
python manage.py runserver
Open a browser and go to http://127.0.0.1:8000 to access the application.

Usage
Upload an Executable File: Navigate to the file upload section in the web app.
Get Classification: The model will analyze the file and display its classification as "Safe" or "Ransomware."
View Previous Results: Access past classifications and analysis results from the database.
