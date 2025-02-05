# Restaurants Analysis
A Comprehensive Analysis of Global Restaurants Using Python

## LEVEL 1
**TASK 1**

**Task: Top Cuisines**
1. Determine the top three most common cuisines in the dataset.
2. Calculate the percentage of restaurants that serve each of the top cuisines.

``` PYTHON
import pandas as pd
# Load dataset
# Load dataset with a different encoding
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')
df
```
**1. Determine the top three most common cuisines in the dataset.**
``` PYTHON
# The 'Cuisines' column contains a string with comma-separated values
# Split the cuisines and explode the list into individual rows
cuisines_split = df['Cuisines'].str.split(',').explode().str.strip()

# Count the occurrences of each cuisine
cuisine_counts = cuisines_split.value_counts()

# Display the top 3 most common cuisines
top_3_cuisines = cuisine_counts.head(3)
print(top_3_cuisines)
```
**Result:**
| Cuisines      | Count |
|--------------|-------|
| North Indian | 3960  |
| Chinese      | 2735  |
| Fast Food    | 1986  |

**2. Calculate the percentage of restaurants that serve each of the top cuisines.**
``` PYTHON
# The 'Cuisines' column contains a string with comma-separated values
# Split the cuisines and explode the list into individual rows
cuisines_split = df['Cuisines'].str.split(',').explode().str.strip()

# Count the occurrences of each cuisine
cuisine_counts = cuisines_split.value_counts()

# Get the top 3 most common cuisines
top_3_cuisines = cuisine_counts.head(3)

# Calculate the percentage of restaurants serving each of the top 3 cuisines
total_restaurants = len(df)
percentage_top_3 = (cuisine_counts.loc[top_3_cuisines.index] / total_restaurants) * 100

# Display the percentage of restaurants serving each of the top 3 cuisines
print(percentage_top_3)
```
**Result:**
| Cuisines      | Percentage |
|--------------|------------|
| North Indian | 41.461627  |
| Chinese      | 28.635745  |
| Fast Food    | 20.793634  |

**TASK 2**

**Task: City Analysis**
1. Identify the city with the highest number of restaurants in the dataset.
2. Calculate the average rating for restaurants in each city.
3. Determine the city with the highest average rating.

**1. Identify the city with the highest number of restaurants in the dataset.**
``` PYTHON
# Count the occurrences of each city
city_counts = df['City'].value_counts()

# Identify the city with the highest number of restaurants
city_with_max_restaurants = city_counts.idxmax()
max_restaurants_count = city_counts.max()

# Display the city with the highest number of restaurants
print(f"The city with the highest number of restaurants is {city_with_max_restaurants} with {max_restaurants_count} restaurants.")
```
**Result:**

The city with the highest number of restaurants is New Delhi with 5473 restaurants.


**2. Calculate the average rating for restaurants in each city.**
``` PYTHON
# Group by 'City' and calculate the average rating for each city
average_rating_by_city = df.groupby('City')['Aggregate rating'].mean()

# Display the average rating for restaurants in each city
print(average_rating_by_city)
```
**Result:**
| City             | Rating  |
|-----------------|---------|
| Abu Dhabi       | 4.300000 |
| Agra           | 3.965000 |
| Ahmedabad      | 4.161905 |
| Albany         | 3.555000 |
| Allahabad      | 3.395000 |
| ...           | ...     |
| Waterloo       | 3.650000 |
| Weirton        | 3.900000 |
| Wellington City | 4.250000 |
| Winchester Bay  | 3.200000 |
| Yorkton        | 3.300000 |

**3. Determine the city with the highest average rating.**
``` PYTHON
# Group by 'City' and calculate the average rating for each city
average_rating_by_city = df.groupby('City')['Aggregate rating'].mean()

# Identify the city with the highest average rating
city_with_highest_rating = average_rating_by_city.idxmax()
highest_rating = average_rating_by_city.max()

# Display the city with the highest average rating
print(f"The city with the highest average rating is {city_with_highest_rating} with an average rating of {highest_rating}.")
```
**Result:**

The city with the highest average rating is Inner City with an average rating of 4.9.

**TASK 3**

**Task: Price Range Distribution**
1. Create a histogram or bar chart to visualize the distribution of price ranges among the restaurants.
2. Calculate the percentage of restaurants in each price range category.

**1. Create a histogram or bar chart to visualize the distribution of price ranges among the restaurants.**
``` PYTHON
import matplotlib.pyplot as plt

# Plot the distribution of price ranges
plt.figure(figsize=(8, 6))
df['Price range'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Distribution of Price Ranges Among Restaurants', fontsize=14)
plt.xlabel('Price Range', fontsize=12)
plt.ylabel('Number of Restaurants', fontsize=12)
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()
```
![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/1.%20Distribution%20of%20Price%20Ranges%20Among%20Restaurants.png)

**2. Calculate the percentage of restaurants in each price range category.**
``` PYTHON
# Calculate the percentage of restaurants in each price range
price_range_counts = df['Price range'].value_counts()
total_restaurants = len(df)

# Calculate the percentage for each price range
price_range_percentage = (price_range_counts / total_restaurants) * 100

# Display the percentage of restaurants in each price range
print(price_range_percentage)
```
**Result:**
| Price Range | Percentage  |
|------------|------------|
| 1          | 46.529159  |
| 2          | 32.593446  |
| 3          | 14.741912  |
| 4          | 6.135483   |

**TASK 4**

**Task: Online Delivery**
1. Determine the percentage of restaurants that offer online delivery.
2. Compare the average ratings of restaurants with and without online delivery

**1. Determine the percentage of restaurants that offer online delivery.**
``` PYTHON
# Convert 'Yes'/'No' values to boolean (True/False)
restaurants_with_online_delivery = (df['Has Online delivery'] == 'Yes').sum()

# Total number of restaurants
total_restaurants = len(df)

# Calculate the percentage of restaurants that offer online delivery
percentage_online_delivery = (restaurants_with_online_delivery / total_restaurants) * 100

# Display the percentage
print(f"The percentage of restaurants that offer online delivery is {percentage_online_delivery:.2f}%.")
```
**Result:**

The percentage of restaurants that offer online delivery is 25.66%.

**2. Compare the average ratings of restaurants with and without online delivery**
``` PYTHON
# Convert 'Yes'/'No' to boolean values for easier comparison
df['Has Online delivery'] = df['Has Online delivery'] == 'Yes'

# Calculate the average rating for restaurants with online delivery
average_rating_with_delivery = df[df['Has Online delivery']].groupby('Has Online delivery')['Aggregate rating'].mean().iloc[0]

# Calculate the average rating for restaurants without online delivery
average_rating_without_delivery = df[~df['Has Online delivery']].groupby('Has Online delivery')['Aggregate rating'].mean().iloc[0]

# Display the average ratings
print(f"Average rating for restaurants with online delivery: {average_rating_with_delivery:.2f}")
print(f"Average rating for restaurants without online delivery: {average_rating_without_delivery:.2f}")
```
**Result:**

Average rating for restaurants with online delivery: 3.25

Average rating for restaurants without online delivery: 2.47

## LEVEL 2
**TASK 1**

**Task: Restaurant Ratings**
1. Analyze the distribution of aggregate ratings and determine the most common rating range.
2. Calculate the average number of votes received by restaurants.

**1. Analyze the distribution of aggregate ratings and determine the most common rating range.**
``` PYTHON
# Plot the distribution of aggregate ratings (histogram)
plt.figure(figsize=(8, 6))
df['Aggregate rating'].hist(bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Distribution of Aggregate Ratings', fontsize=14)
plt.xlabel('Aggregate Rating', fontsize=12)
plt.ylabel('Number of Restaurants', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Define rating ranges (e.g., 0-1, 1-2, 2-3, etc.)
rating_bins = [0, 1, 2, 3, 4, 5]
df['Rating range'] = pd.cut(df['Aggregate rating'], bins=rating_bins, right=False)

# Determine the most common rating range
most_common_rating_range = df['Rating range'].value_counts().idxmax()

# Display the most common rating range
print(f"The most common rating range is {most_common_rating_range}.")
```
**Result:**

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/2.%20Distribution%20of%20Aggregate%20Ratings.png)

The most common rating range is [3, 4).

**2. Calculate the average number of votes received by restaurants.**
``` PYTHON
# Calculate the average number of votes received by restaurants
average_votes = df['Votes'].mean()

# Display the result
print(f"The average number of votes received by restaurants is {average_votes:.2f}.")
```
**Result:**

The average number of votes received by restaurants is 156.91.

**TASK 2**

**Task: Cuisine Combination**
1. Identify the most common combinations of cuisines in the dataset.
2. Determine if certain cuisine combinations tend to have higher ratings.

**1. Identify the most common combinations of cuisines in the dataset.**
``` PYTHON
from collections import Counter

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Split the 'Cuisines' column by ',' to handle multiple cuisines
df['Cuisines'] = df['Cuisines'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# Convert the list of cuisines into tuples (to make them hashable for counting)
cuisine_combinations = df['Cuisines'].apply(lambda x: tuple(sorted(x)))

# Count the occurrences of each cuisine combination
combination_counts = Counter(cuisine_combinations)

# Get the most common cuisine combinations
most_common_combinations = combination_counts.most_common()

# Display the most common combinations
for combination, count in most_common_combinations:
    print(f"Combination: {combination}, Count: {count}")
```
**Result:**

Combination: ('North Indian',), Count: 936

Combination: ('Chinese', 'North Indian'), Count: 616

Combination: ('Mughlai', 'North Indian'), Count: 394

Combination: ('Chinese',), Count: 354

Combination: ('Fast Food',), Count: 354

Combination: ('Chinese', 'Mughlai', 'North Indian'), Count: 306

.........................

.........................

**2. Determine if certain cuisine combinations tend to have higher ratings.**
``` PYTHON
import pandas as pd
from collections import Counter

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Split the 'Cuisines' column by ',' to handle multiple cuisines
df['Cuisines'] = df['Cuisines'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# Convert the list of cuisines into tuples (to make them hashable for grouping)
df['Cuisine Combination'] = df['Cuisines'].apply(lambda x: tuple(sorted(x)))

# Group by cuisine combinations and calculate the average rating for each combination
average_ratings_by_combination = df.groupby('Cuisine Combination')['Aggregate rating'].mean()

# Sort the combinations by average rating in descending order
sorted_combinations = average_ratings_by_combination.sort_values(ascending=False)

# Display the top cuisine combinations with higher average ratings
print("Cuisine combinations with higher average ratings:")
print(sorted_combinations)
```
**Result:**
| Cuisine Combination                                                   | Average Rating |
|------------------------------------------------------------------------|---------------|
| (World Cuisine,)                                                      | 4.9           |
| (American, BBQ, Sandwich)                                             | 4.9           |
| (American, Sandwich, Tea)                                             | 4.9           |
| (Indonesian, Sunda)                                                   | 4.9           |
| (Hawaiian, Seafood)                                                   | 4.9           |
| ...                                                                    | ...           |
| (Continental, Fast Food, Italian, North Indian)                       | 0.0           |
| (Continental, Fast Food, North Indian, South Indian, Street Food)     | 0.0           |
| (Bakery, Chinese, Mithai, North Indian, South Indian, Street Food)    | 0.0           |
| (Biryani, Fast Food, Healthy Food, Pizza)                             | 0.0           |
| (Lucknowi, Mughlai, North Indian)                                     | 0.0           |

``` PYTHON
# Filter out cuisine combinations with an average rating of 0.0
filtered_combinations = sorted_combinations[sorted_combinations > 0.0]

# Display the top cuisine combinations with higher ratings
print("Cuisine combinations with higher average ratings (filtered):")
print(filtered_combinations.head())  # Display the top 5 combinations
```
**Result:**
| Cuisine Combination            | Average Rating |
|--------------------------------|---------------|
| (World Cuisine,)               | 4.9           |
| (American, BBQ, Sandwich)      | 4.9           |
| (American, Sandwich, Tea)      | 4.9           |
| (Indonesian, Sunda)            | 4.9           |
| (Hawaiian, Seafood)            | 4.9           |

**TASK 3**

**Task: Geographic Analysis**
1. Plot the locations of restaurants on a map using longitude and latitude coordinates.
2. Identify any patterns or clusters of restaurants in specific areas.

**1. Plot the locations of restaurants on a map using longitude and latitude coordinates.**
``` PYTHON
import folium

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Create a base map centered around a specific location (e.g., the average location of all restaurants)
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

# Loop through each row and add a marker to the map
for index, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(restaurant_map)

# Save the map to an HTML file
restaurant_map.save('restaurant_map.html')

# Display the map
restaurant_map
```

**2. Identify any patterns or clusters of restaurants in specific areas.**
``` PYTHON
import pandas as pd
import folium
from sklearn.cluster import KMeans
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Ensure your dataset has the 'Latitude' and 'Longitude' columns
coordinates = df[['Latitude', 'Longitude']]

# Apply KMeans clustering to find clusters
kmeans = KMeans(n_clusters=5, random_state=42)  # You can change n_clusters to fit your needs
df['Cluster'] = kmeans.fit_predict(coordinates)

# Create a base map centered around the average location
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

# Define a color palette for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Loop through each row and add a marker with a cluster color
for index, row in df.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=row['Restaurant Name'],
        icon=folium.Icon(color=colors[row['Cluster']])
    ).add_to(restaurant_map)

# Save the map to an HTML file
restaurant_map.save('clustered_restaurant_map.html')

# Display the map
restaurant_map
```

**TASK 4**

**Task: Restaurant Chains**
1. Identify if there are any restaurant chains present in the dataset.
2. Analyze the ratings and popularity of different restaurant chains.

**1. Identify if there are any restaurant chains present in the dataset.**
``` PYTHON
# Load dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Count the occurrences of each restaurant name
restaurant_counts = df['Restaurant Name'].value_counts()

# Filter out the restaurants that appear more than once
restaurant_chains = restaurant_counts[restaurant_counts > 1]

# Display the restaurant chains
print("Restaurant Chains found in the dataset:")
print(restaurant_chains)
```
**Result:**
| Restaurant Name         | Count |
|------------------------|-------|
| Cafe Coffee Day       | 83    |
| Domino's Pizza        | 79    |
| Subway               | 63    |
| Green Chick Chop     | 51    |
| McDonald's          | 48    |
| ...                  | ...   |
| Garota de Ipanema    | 2     |
| Super Snacks        | 2     |
| Qureshi Kabab Corner | 2     |
| Silantro Fil-Mex    | 2     |
| Harry's Bar + Cafe  | 2     |

**2. Analyze the ratings and popularity of different restaurant chains.**
``` PYTHON
# Clean the restaurant names if necessary (e.g., handling variations like apostrophes or case sensitivity)
df['Restaurant Name'] = df['Restaurant Name'].str.replace("'", "").str.lower()

# Group by Restaurant Name and calculate the average rating and total votes
chain_analysis = df.groupby('Restaurant Name').agg(
    average_rating=('Aggregate rating', 'mean'),
    total_votes=('Votes', 'sum')
)

# Sort by average rating and total votes
chain_analysis_sorted_by_rating = chain_analysis.sort_values(by='average_rating', ascending=False)
chain_analysis_sorted_by_popularity = chain_analysis.sort_values(by='total_votes', ascending=False)

# Display the top 10 restaurant chains by rating and popularity
print("Top 10 Restaurant Chains by Average Rating:")
print(chain_analysis_sorted_by_rating.head(10))

print("\nTop 10 Restaurant Chains by Popularity (Total Votes):")
print(chain_analysis_sorted_by_popularity.head(10))
```
**Result:**
**Top 10 Restaurant Chains by Average Rating**

| Restaurant Name                                      | Average Rating | Total Votes |
|-----------------------------------------------------|---------------|------------|
| bao                                               | 4.9           | 161        |
| spiral - sofitel philippine plaza manila          | 4.9           | 621        |
| ingleside village pizza                            | 4.9           | 478        |
| tantra asian bistro                                | 4.9           | 474        |
| mr. dunderbaks biergarten and marketplatz         | 4.9           | 1413       |
| draft gastro pub                                   | 4.9           | 522        |
| yellow dog eats                                    | 4.9           | 1252       |
| talaga sampireun                                   | 4.9           | 5514       |
| gaga manjero                                       | 4.9           | 95         |
| cube - tasting kitchen                            | 4.9           | 441        |

**Top 10 Restaurant Chains by Popularity (Total Votes)**

| Restaurant Name            | Average Rating | Total Votes |
|----------------------------|---------------|------------|
| barbeque nation           | 4.35          | 28142      |
| abs - absolute barbecues  | 4.83          | 13400      |
| toit                      | 4.80          | 10934      |
| big chill                 | 4.48          | 10853      |
| farzi cafe                | 4.37          | 10098      |
| truffles                  | 3.95          | 9682       |
| chilis                    | 4.58          | 8156       |
| hauz khas social          | 4.30          | 7931       |
| joeys pizza               | 4.25          | 7807       |
| peter cat                 | 4.30          | 7574       |

``` PYTHON
import matplotlib.pyplot as plt

# Top 10 restaurant chains by rating
top_rated_chains = chain_analysis_sorted_by_rating.head(10)

# Plot the average ratings
plt.figure(figsize=(10, 6))
top_rated_chains['average_rating'].plot(kind='bar', color='skyblue')
plt.title('Top 10 Restaurant Chains by Average Rating')
plt.ylabel('Average Rating')
plt.xlabel('Restaurant Name')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Top 10 restaurant chains by popularity (votes)
top_popular_chains = chain_analysis_sorted_by_popularity.head(10)

# Plot the total votes
plt.figure(figsize=(10, 6))
top_popular_chains['total_votes'].plot(kind='bar', color='salmon')
plt.title('Top 10 Restaurant Chains by Popularity (Total Votes)')
plt.ylabel('Total Votes')
plt.xlabel('Restaurant Name')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/3.%20Top%2010%20Restaurant%20Chains%20By%20Average%20Rating.png)

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/4.%20Top%2010%20Restaurants%20Chains%20By%20Popularity%20(Total%20Votes).png)

## LEVEL 3

**TASK 1**

**Task: Restaurant Reviews**
1. Analyze the text reviews to identify the most common positive and negative keywords.
2. Calculate the average length of reviews and explore if there is a relationship between review length and rating.

**1. Analyze the text reviews to identify the most common positive and negative keywords.**
``` PYTHON
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Clean and preprocess the reviews (remove special characters, lowercase, etc.)
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['Cleaned_Reviews'] = df['Rating text'].apply(clean_text)

# Perform sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify as positive (1), neutral (0), or negative (-1)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['Sentiment'] = df['Cleaned_Reviews'].apply(get_sentiment)

# Separate the reviews into positive and negative reviews
positive_reviews = df[df['Sentiment'] == 'positive']['Cleaned_Reviews']
negative_reviews = df[df['Sentiment'] == 'negative']['Cleaned_Reviews']

# Function to extract the most common words using CountVectorizer
def get_most_common_words(reviews):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), max_features=20, ngram_range=(1, 2))  # Unigrams and bigrams
    word_matrix = vectorizer.fit_transform(reviews)
    word_count = word_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    word_freq = dict(zip(words, word_count))
    return word_freq

# Get most common words for positive and negative reviews
positive_keywords = get_most_common_words(positive_reviews)
negative_keywords = get_most_common_words(negative_reviews)

# Plot the most common positive keywords
plt.figure(figsize=(10, 6))
plt.barh(list(positive_keywords.keys()), list(positive_keywords.values()), color='green')
plt.title('Most Common Positive Keywords in Reviews')
plt.xlabel('Frequency')
plt.ylabel('Keywords')
plt.show()

# Plot the most common negative keywords
plt.figure(figsize=(10, 6))
plt.barh(list(negative_keywords.keys()), list(negative_keywords.values()), color='red')
plt.title('Most Common Negative Keywords in Reviews')
plt.xlabel('Frequency')
plt.ylabel('Keywords')
plt.show()

# Optionally, generate word clouds for a more visual representation
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_keywords)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(negative_keywords)

# Display word clouds
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Review Word Cloud')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Review Word Cloud')
plt.axis('off')
plt.show()
```
**Result:**

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/5.%20Most%20Common%20Positive%20Keywords%20In%20Reviews.png)

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/6.%20Most%20Negative%20Keywords%20In%20Reviews.png)

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/7.%20Positive%20Review%20Word%20Cloud.png)

**2. Calculate the average length of reviews and explore if there is a relationship between review length and rating.**
``` PYTHON
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Clean the reviews column (remove any missing or NaN values)
df['Cleaned_Reviews'] = df['Rating text'].dropna()

# Calculate the length of each review in terms of characters or words
df['Review_Length'] = df['Cleaned_Reviews'].apply(lambda x: len(x.split()))  # Length in terms of words
# Alternatively, use len(x) for character length: df['Review_Length'] = df['Cleaned_Reviews'].apply(len)

# Calculate the average length of reviews
average_review_length = df['Review_Length'].mean()
print(f"Average Review Length (in words): {average_review_length}")

# Explore the relationship between review length and rating
plt.figure(figsize=(10, 6))

# Scatter plot: Review length vs. Rating
sns.scatterplot(x=df['Review_Length'], y=df['Aggregate rating'], color='blue')
plt.title('Review Length vs. Aggregate Rating')
plt.xlabel('Review Length (in words)')
plt.ylabel('Aggregate Rating')
plt.show()

# Calculate correlation between review length and rating
correlation = df[['Review_Length', 'Aggregate rating']].corr()
print("Correlation between Review Length and Rating:")
print(correlation)

# Optional: Plot a heatmap to visualize the correlation
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap: Review Length vs Rating')
plt.show()
```
**Result:**

Average Review Length (in words): 1.3378703800649148

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/8.%20Review%20Length%20vs%20Aggregate%20Rating.png)

Correlation between Review Length and Rating

|                     | Review Length | Aggregate Rating |
|---------------------|---------------|------------------|
| **Review Length**    | 1.000000      | -0.599573        |
| **Aggregate Rating** | -0.599573     | 1.000000         |

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/9.%20Correlation%20Heatmap%20Review%20Length%20vs%20Rating.png)

**TASK 2**

**Task: Votes Analysis**
1. Identify the restaurants with the highest and lowest number of votes.
2. Analyze if there is a correlation between the number of votes and the rating of a restaurant.

**1. Identify the restaurants with the highest and lowest number of votes.**
``` PYTHON
import pandas as pd

# Load dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Identify the restaurant with the highest number of votes
highest_votes = df.loc[df['Votes'].idxmax()]

# Identify the restaurant with the lowest number of votes
lowest_votes = df.loc[df['Votes'].idxmin()]

# Display the restaurants with the highest and lowest number of votes
print(f"Restaurant with the Highest Number of Votes: {highest_votes['Restaurant Name']}")
print(f"Votes: {highest_votes['Votes']}")

print(f"\nRestaurant with the Lowest Number of Votes: {lowest_votes['Restaurant Name']}")
print(f"Votes: {lowest_votes['Votes']}")
```
**Result:**

Restaurant with the Highest Number of Votes: Toit
Votes: 10934

Restaurant with the Lowest Number of Votes: Cantinho da Gula
Votes: 0

**2. Analyze if there is a correlation between the number of votes and the rating of a restaurant.**
``` PYTHON
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Calculate the correlation between the number of votes and aggregate rating
correlation = df[['Votes', 'Aggregate rating']].corr()

# Display the correlation matrix
print("Correlation between Votes and Aggregate Rating:")
print(correlation)

# Scatter plot: Votes vs. Aggregate Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Votes'], y=df['Aggregate rating'], color='purple')
plt.title('Votes vs. Aggregate Rating')
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate Rating')
plt.show()

# Optional: Calculate Pearson correlation coefficient directly
pearson_corr = df['Votes'].corr(df['Aggregate rating'])
print(f"Pearson Correlation Coefficient: {pearson_corr:.2f}")
```
**Result:**

Correlation between Votes and Aggregate Rating

|                     | Votes        | Aggregate Rating |
|---------------------|--------------|------------------|
| **Votes**           | 1.000000     | 0.313691         |
| **Aggregate Rating**| 0.313691     | 1.000000         |

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/10.%20Votes%20vs%20Aggregate%20Rating.png)

Pearson Correlation Coefficient: 0.31

**TASK 3**

**Task: Price Range vs. Online Delivery andTable Booking**
1. Analyze if there is a relationship between the price range and the availability of online delivery and table booking.
2. Determine if higher-priced restaurants are more likely to offer these services.

**1. Analyze if there is a relationship between the price range and the availability of online delivery and table booking.**
``` PYTHON
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Convert 'Has Table booking' and 'Has Online delivery' to binary values (1 for 'Yes' and 0 for 'No')
df['Has Table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

# Group by Price range and calculate the proportion of restaurants offering online delivery and table booking
price_range_delivery_booking = df.groupby('Price range')[['Has Table booking', 'Has Online delivery']].mean()

# Display the results
print(price_range_delivery_booking)

# Visualize the availability of table booking and online delivery for each price range using a bar plot
price_range_delivery_booking.plot(kind='bar', figsize=(10, 6), stacked=True)
plt.title('Availability of Online Delivery and Table Booking by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Proportion of Restaurants')
plt.xticks(rotation=0)
plt.legend(title='Service Availability', labels=['Table Booking', 'Online Delivery'])
plt.show()

# Optional: Perform a chi-squared test to check for independence (if necessary)
from scipy.stats import chi2_contingency

# Create contingency table for chi-squared test
contingency_table = pd.crosstab(df['Price range'], df['Has Online delivery'])
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

# Display the chi-squared test result
print(f"Chi-Squared Test result for Price Range vs. Online Delivery:\nStatistic: {chi2_stat}, P-value: {p_val}")
```
**Result:**

Has Table Booking and Has Online Delivery by Price Range

| Price Range | Has Table Booking | Has Online Delivery |
|-------------|-------------------|---------------------|
| 1           | 0.000225          | 0.157741            |
| 2           | 0.076775          | 0.413106            |
| 3           | 0.457386          | 0.291903            |
| 4           | 0.467577          | 0.090444            |

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/11.%20Availability%20of%20Online%20Delivery%20and%20Table%20Booking%20by%20Price%20Range.png)

Chi-Squared Test result for Price Range vs. Online Delivery:

Statistic: 721.3786767489615, P-value: 4.855491091732406e-156

**2. Determine if higher-priced restaurants are more likely to offer these services.**
``` PYTHON
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/content/drive/MyDrive/CogniFy Technologies/Dataset CSV.csv', encoding='ISO-8859-1')

# Convert 'Has Table booking' and 'Has Online delivery' to binary values (1 for 'Yes' and 0 for 'No')
df['Has Table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

# Group by Price range and calculate the proportion of restaurants offering online delivery and table booking
price_range_service_availability = df.groupby('Price range')[['Has Table booking', 'Has Online delivery']].mean()

# Display the proportions of service availability by price range
print(price_range_service_availability)

# Visualize the service availability for each price range using a bar plot
price_range_service_availability.plot(kind='bar', figsize=(10, 6), stacked=True)
plt.title('Availability of Online Delivery and Table Booking by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Proportion of Restaurants')
plt.xticks(rotation=0)
plt.legend(title='Service Availability', labels=['Table Booking', 'Online Delivery'])
plt.show()

# Additional Analysis: Compare proportions for higher-priced vs. lower-priced
# Defining higher and lower price ranges based on median or any chosen threshold
median_price_range = df['Price range'].median()

# Separate into higher and lower price range categories
df['Price category'] = df['Price range'].apply(lambda x: 'Higher' if x >= median_price_range else 'Lower')

# Calculate the mean proportion of services for higher and lower price categories
price_category_service_availability = df.groupby('Price category')[['Has Table booking', 'Has Online delivery']].mean()

# Display the results for higher vs. lower price categories
print(price_category_service_availability)
```
**Result:**

Has Table Booking and Has Online Delivery by Price Range

| Price Range | Has Table Booking | Has Online Delivery |
|-------------|-------------------|---------------------|
| 1           | 0.000225          | 0.157741            |
| 2           | 0.076775          | 0.413106            |
| 3           | 0.457386          | 0.291903            |
| 4           | 0.467577          | 0.090444            |

![](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/11b.%20Availability%20of%20Online%20Delivery%20and%20Table%20Booking%20by%20Price%20Range.png)

Has Table Booking and Has Online Delivery by Price Category

| Price Category | Has Table Booking | Has Online Delivery |
|----------------|-------------------|---------------------|
| Higher         | 0.226552          | 0.342667            |
| Lower          | 0.000225          | 0.157741            |














