import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Netflix Dataset...")
try:
   
    df = pd.read_csv('netflix_titles.csv')
    print(f"Data successfully! Total rows: {len(df)}\n")
except FileNotFoundError:
    print("Error: 'netflix_titles.csv' not found.")
    # Initialize df as an empty DataFrame to prevent NameError and allow script to continue
    df = pd.DataFrame({'country': [], 'cast': [], 'director': [], 'date_added': [], 'rating': [], 'duration': [], 'type': [], 'year_added': [], 'listed_in': [], 'description': [], 'title': [], 'release_year': []})
    print("Continuing execution with an empty DataFrame. Please upload 'netflix_titles.csv' to run the analysis properly.")

# ==========================================
# PART 1: DATA CLEANING & PREPROCESSING
# ==========================================
print("--- Step 1: Data Cleaning ---")

df['country'] = df['country'].fillna('Unknown')
df['cast'] = df['cast'].fillna('No Cast')
df['director'] = df['director'].fillna('Unknown Director')
df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)

# ==========================================
# PART 2: ADVANCED DATA ANALYTICS (EDA)
# ==========================================
print("--- Step 2: Advanced Analytics Start ---")
sns.set_theme(style="whitegrid")

# 1. Movies vs TV Shows (Pie Chart) - 
plt.figure(figsize=(7, 7))
if not df.empty:
    df['type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, explode=(0.05, 0))
    plt.title('Distribution of Content: Movies vs TV Shows', fontsize=14)
    plt.ylabel('')
    plt.savefig(f'{output_dir}/1_content_distribution_pie.png')
    plt.close()
else:
    print("Skipping 'Movies vs TV Shows' plot: DataFrame is empty.")

# 2. Content Growth (Area Plot) - 
plt.figure(figsize=(12, 6))
if not df.empty:
    df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year
    year_growth = df['year_added'].value_counts().sort_index()
    plt.fill_between(year_growth.index, year_growth.values, color="skyblue", alpha=0.4)
    plt.plot(year_growth.index, year_growth.values, color="Slateblue", alpha=0.6, linewidth=2)
    plt.title('Netflix Content Addition Trend (Area Chart)', fontsize=15)
    plt.xlim(2008, 2021)
    plt.savefig(f'{output_dir}/2_content_growth_trend.png')
    plt.close()
else:
    print("Skipping 'Content Growth' plot: DataFrame is empty.")

# 3. Top 10 Genres - 
plt.figure(figsize=(10, 6))
if not df.empty:
    df['primary_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0] if pd.notna(x) else 'Unknown')
    top_genres = df['primary_genre'].value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='rocket')
    plt.title('Top 10 Performing Genres on Netflix', fontsize=15)
    plt.savefig(f'{output_dir}/3_top_genres.png')
    plt.close()
else:
    print("Skipping 'Top 10 Genres' plot: DataFrame is empty.")

# 4. Statistical Insight: Movie Runtimes (Skewness) - 
if not df.empty and 'type' in df.columns and 'duration' in df.columns:
    movies_df = df[df['type'] == 'Movie'].copy()
    if not movies_df.empty:
        movies_df['duration_min'] = movies_df['duration'].str.replace(' min', '', regex=False).astype(float)
        if not movies_df['duration_min'].empty:
            data_skewness = skew(movies_df['duration_min'])

            plt.figure(figsize=(10, 5))
            sns.kdeplot(movies_df['duration_min'], fill=True, color="olive")
            plt.title(f'Movie Runtime Distribution (Skewness: {data_skewness:.2f})', fontsize=14)
            plt.xlabel('Duration in Minutes')
            plt.savefig(f'{output_dir}/4_runtime_stats.png')
            plt.close()
        else:
            print("Skipping 'Movie Runtimes' plot: No movie durations found.")
    else:
        print("Skipping 'Movie Runtimes' plot: No movies in DataFrame.")
else:
    print("Skipping 'Movie Runtimes' plot: DataFrame is empty or missing necessary columns.")

# 5. Top 10 Content Producing Countries -
plt.figure(figsize=(12, 6))
if not df.empty:
    df['first_country'] = df['country'].apply(lambda x: x.split(",")[0] if pd.notna(x) else 'Unknown')
    top_countries = df['first_country'].value_counts().head(10)
    if 'Unknown' in top_countries: top_countries = top_countries.drop('Unknown')
    sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Top 10 Content Producing Countries', fontsize=15)
    plt.savefig(f'{output_dir}/5_global_production.png')
    plt.close()
else:
    print("Skipping 'Top 10 Content Producing Countries' plot: DataFrame is empty.")

# ==========================================
# PART 3: MACHINE LEARNING (Random Forest)
# ==========================================
print("\n--- Step 3: Machine Learning Model ---")

if not df.empty and all(col in df.columns for col in ['release_year', 'description', 'title', 'listed_in', 'type']):
    # Feature Engineering: 
    df['desc_words'] = df['description'].apply(lambda x: len(str(x).split()))
    df['title_len'] = df['title'].apply(lambda x: len(str(x)))
    df['genre_count'] = df['listed_in'].apply(lambda x: len(str(x).split(',')))

    # X = Input Features, y = Target Output (Movie=0, TV Show=1)
    X = df[['release_year', 'desc_words', 'title_len', 'genre_count']]
    y = df['type'].map({'Movie': 0, 'TV Show': 1})

    # Check if there's enough data for training after mapping
    if len(X) > 1 and len(y.unique()) > 1:
        # Training/Testing Split (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model initialization
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Training the classification model...")
        model.fit(X_train, y_train)

        # Accuracy Calculation
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Classification Accuracy: {score * 100:.2f}%")

        # Confusion Matrix Visualisation
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=['Movie', 'TV Show'], yticklabels=['Movie', 'TV Show'])
        plt.title('Model Performance: Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'{output_dir}/6_ml_results.png')
        plt.close()
    else:
        print("Skipping Machine Learning Model: Not enough data or target classes for training after processing.")
else:
    print("Skipping Machine Learning Model: DataFrame is empty or missing necessary columns for feature engineering.")

print("\n🎉 Project Execution Complete! Graphs are saved in 'visualizations/' folder.")
