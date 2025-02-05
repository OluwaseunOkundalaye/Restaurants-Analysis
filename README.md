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
**Result:**

![Map](https://github.com/OluwaseunOkundalaye/Restaurants-Analysis/blob/main/Map%201.html)


















