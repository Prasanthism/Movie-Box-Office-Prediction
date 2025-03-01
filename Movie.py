import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:\\Users\\prasa\\Downloads\\TMDB_movie_dataset_v11.csv")
df.fillna({'budget': 0, 'revenue': 0, 'popularity': 0, 'vote_average': 0, 'vote_count': 0}, inplace=True)

def extract_main_genre(genres):
    try:
        return eval(genres)[0]['name'] if eval(genres) else 'Unknown'
    except:
        return 'Unknown'

df['main_genre'] = df['genres'].apply(extract_main_genre)
features = ['budget', 'popularity', 'vote_average', 'vote_count', 'main_genre']
df = df[features + ['revenue', 'title']]

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[['main_genre']])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())

scaler = StandardScaler()
numerical_features = df[['budget', 'popularity', 'vote_average', 'vote_count']]
numerical_scaled = scaler.fit_transform(numerical_features)
numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features.columns)
final_df = pd.concat([numerical_df, categorical_df], axis=1)

y = df['revenue']
X = final_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_movie_revenue(movie_title):
    movie_data = df.loc[df['title'].str.lower() == movie_title.lower()]
    if movie_data.empty:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return

    movie_features = movie_data[['budget', 'popularity', 'vote_average', 'vote_count', 'main_genre']]
    genre_encoded = encoder.transform(movie_features[['main_genre']])
    genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out())

    num_features = movie_features[['budget', 'popularity', 'vote_average', 'vote_count']]
    num_scaled = scaler.transform(num_features)
    num_df = pd.DataFrame(num_scaled, columns=num_features.columns)

    movie_final = pd.concat([num_df, genre_df], axis=1)
    predicted_revenue = model.predict(movie_final)[0]

    budget = movie_data['budget'].values[0]
    popularity = movie_data['popularity'].values[0]
    vote_avg = movie_data['vote_average'].values[0]
    vote_count = movie_data['vote_count'].values[0]
    main_genre = movie_data['main_genre'].values[0]

    print(f"\n📌 **Movie Details:** {movie_title.title()}")
    print(f"🔹 Budget: ${budget:,.2f}")
    print(f"🔹 Popularity: {popularity}")
    print(f"🔹 Vote Average: {vote_avg}")
    print(f"🔹 Vote Count: {vote_count}")
    print(f"🔹 Main Genre: {main_genre}")
    print(f"🎬 **Predicted Revenue:** ${predicted_revenue:,.2f}\n")

    popularity_scaled = popularity * 1e6 
    vote_avg_scaled = vote_avg * 50e6 
    vote_count_scaled = vote_count * 1000

    labels = ['Budget', 'Popularity (Scaled)', 'Vote Average (Scaled)', 'Vote Count (Scaled)', 'Predicted Revenue']
    values = [budget, popularity_scaled, vote_avg_scaled, vote_count_scaled, predicted_revenue]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, hue=labels, dodge=False, palette="Blues_d", legend=False)
    plt.title(f"Revenue Prediction for '{movie_title.title()}'", fontsize=14, fontweight='bold')
    plt.ylabel("Scaled Value", fontsize=12)
    plt.xticks(rotation=30, fontsize=10)

    original_values = [budget, popularity, vote_avg, vote_count, predicted_revenue]
    for index, (val, orig_val) in enumerate(zip(values, original_values)):
        plt.text(index, val + (val * 0.05), f"{orig_val:,.2f}", ha='center', fontsize=10, fontweight='bold')
    plt.show()
  
movie_name = input("Enter a movie name: ")
predict_movie_revenue(movie_name)
