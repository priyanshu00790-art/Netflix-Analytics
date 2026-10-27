🎬 Netflix Data Science & Analytics Project 🤖

Python Machine Learning Data Analysis Status

    An integrated study of Exploratory Data Analytics (EDA) and Machine Learning using the official Netflix Movies and TV Shows dataset. Submitted for the "Innovative IT" lab evaluation, this project demonstrates how data-driven insights can help streaming platforms understand content trends and automate categorization.

👥 Team Members
Name 	Enrollment ID
Ravi Singh 	202410101360098
Aman Singh 	202410101360097
Priyanshu Gupta 	202410101360095
Divyanshu Shukla 	202410101360088
📝 Project Overview

The project is divided into two main phases:

    Analytics Phase: Analyzing historical data to find patterns in content growth, geographical dominance, and genre popularity.
    Data Science Phase: Building a predictive model to classify content based on metadata.

🎯 Objectives

    ✅ Data Cleaning: Perform comprehensive data cleaning and handle missing values in the Netflix library.
    ✅ Time-Series Analysis: Visualize the growth of Netflix over the past decade.
    ✅ User Preferences: Understand audience preferences by analyzing maturity ratings and genre distributions.
    ✅ Predictive Modeling: Implement a Machine Learning pipeline using the RandomForestClassifier for automated content classification.

📂 Dataset Description

The dataset consists of metadata for approximately 8,000+ movies and TV shows available on Netflix.
Feature 	Description
show_id 	Unique identifier for each show
type 	Category of content (Movie or TV Show)
title 	Name of the content
country 	Country of production
release_year 	Actual release year of the content
rating 	Maturity rating (e.g., TV-MA, PG-13)
listed_in 	Genres associated with the content
🛠️ Tech Stack & Concepts

    Language: Python 3.x
    Data Manipulation: Pandas, NumPy
    Data Visualization: Matplotlib, Seaborn
    Machine Learning: Scikit-Learn (Random Forest, Train-Test Split)
    Project Management: GitHub for Version Control

📊 Analytics Methodology & Insights

Our EDA produced 6 key visualizations:

    Movies vs. TV Shows: A comparison showing the dominance of feature films over series.
    Yearly Content Trends: A line graph tracking the rapid expansion of Netflix since 2015.
    Genre Breakdown: Analysis of top-performing categories like International Movies and Dramas.
    Duration Analysis: Histogram showing the average runtime of movies (most falling between 90-120 minutes).
    Global Production: A bar chart of the top 10 countries, with the USA and India leading content production.
    ML Performance: A Confusion Matrix visualizing the model's accuracy and classification errors.

🤖 Machine Learning Implementation

We utilized a Supervised Learning approach to solve a classification problem:

    Feature Engineering: Created new features like description_length and num_genres to improve model accuracy.
    Training Process: 80% of the data was used for training the Random Forest model.
    Evaluation: 20% of the data was reserved for testing, ensuring the model generalizes well to new, unseen content.

🚀 How to Run the Project

1. Clone the Repository

git clone [Your-Repo-Link]
🚀 How to Run the Project

Clone the Repository:

git clone [Your-Repo-Link]


Install Dependencies:

pip install -r requirements.txt


Dataset Setup:
Ensure netflix_titles.csv is present in the root directory.

Execution:
Run the analysis script:

python netflix_analytics.py


Output:
Check the visualizations/ folder for all generated plots and the console for the ML Accuracy Score.

📁 Conclusion

This project successfully demonstrates the power of Data Science in the entertainment industry. Through EDA, we identified key growth regions and popular content types, while the ML model provided a baseline for automated content management.

