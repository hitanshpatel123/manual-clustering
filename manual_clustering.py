import pandas as pd
import seaborn as sns
from seaborn import scatterplot
import matplotlib.pyplot as plt

customers = pd.read_csv('mall_customers.csv')
print(customers.head())
print(customers.describe())
print(customers.shape)
print(customers.isna().sum())
correlations = customers.select_dtypes(include='number').corr()
print(correlations)


cols_to_keep = ['Age', 'Spending Score']
customers = customers[cols_to_keep]
centroids = customers.sample(2)

def fetch_coordinates(df):
    age_centroid_1 = df.iloc[0, 0]
    score_centroid_1 = df.iloc[0, 1]
    age_centroid_2 = df.iloc[1, 0]
    score_centroid_2 = df.iloc[1, 1]
    return age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2

age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2 =fetch_coordinates(centroids)

figure , ax = plt.subplots()
ax.scatter(customers['Age'],customers['Spending Score'])
ax.scatter(centroids['Age'],centroids['Spending Score'], s = 100, c= 'Black')
plt.show()

def calculate_distance(row, coord_1, coord_2):
    dist = np.sqrt((row[0]-coord_1)**2 + (row[1]-coord_2)**2)
    return dist

customers['dist_centroid_1'] = customers.apply(calculate_distance, args = [age_centroid_1, score_centroid_1], axis = 1)
customers['dist_centroid_2'] = customers.apply(calculate_distance, args = [age_centroid_2, score_centroid_2], axis = 1)


def calculate_distance_assign_clusters(customers, centroids):
    age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2 = fetch_coordinates(centroids)
    customers['dist_centroid_1'] = customers.apply(calculate_distance, args = [age_centroid_1, score_centroid_1], axis = 1)
    customers['dist_centroid_2'] = customers.apply(calculate_distance, args = [age_centroid_2, score_centroid_2], axis = 1)
    
    customers['cluster'] = np.where(customers['dist_centroid_1']<customers['dist_centroid_2'],1,2)
    
    return customers

customers = calculate_distance_assign_clusters(customers, centroids)

figure, ax = plt.subplots()
sns.scatterplot(data= customers, x = 'Age', y= 'Spending Score', hue= 'cluster', palette = 'tab10')
sns.scatterplot(data=centroids, x = 'Age', y= 'Spending Score', s=100)
plt.show()

new_centroids = customers.groupby('cluster').mean().drop(['dist_centroid_1','dist_centroid_2'], axis=1)
new_centroids = new_centroids.reset_index()
new_centroids = new_centroids.drop('cluster', axis = 1)

customers = calculate_distance_assign_clusters(customers, new_centroids)

figure , ax= plt.subplots()
sns.scatterplot(data=customers, x= 'Age', y='Spending Score', hue='cluster', palette = 'tab10')
sns.scatterplot(data=new_centroids, x= 'Age', y='Spending Score', s=100)
plt.show()

#####################################################################################
def create_clusters(customers):
    centroids = customers.sample(2)
    customers = calculate_distance_assign_clusters(customers,centroids)
    new_centroids = customers.groupby('cluster').mean().reset_index()
    new_centroids = new_centroids[cols_to_keep]
    
    customers = calculate_distance_assign_clusters(customers,new_centroids)
    
    return customers['cluster']

clusters = create_clusters(customers)