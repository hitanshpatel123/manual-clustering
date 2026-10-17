import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

customers = pd.read_csv('mall_customers.csv')

cols_to_keep = ['Annual Income', 'Spending Score']
customers = customers[cols_to_keep].copy()

def get_centroids(data, k):
    cent = data.sample(k).reset_index().drop('index', axis =1)
    coor = cent.values.tolist()
        
    return cent, coor

def calculate_distance(data, coor):
    dist_names = []
    for i, centroid in enumerate(coor):
        name = f'dist_centroid_{i + 1}'
        data[name] = np.sqrt((data.iloc[:, 0] - centroid[0])**2 + (data.iloc[:, 1] - centroid[1])**2)
        dist_names.append(name)
    return data, dist_names

def kmeans(data, k, n_iterations = 100, plot = False):
    variables = data.columns
    centroids, coords = get_centroids(data, k)
    for i in range(n_iterations): 
        last_coords = coords.copy()

        data, dist = calculate_distance(data, coords)

        data['cluster'] = data[dist].idxmin(axis=1).str.split('_').str[-1]

        centroids = round(data.groupby('cluster')[variables].mean(), 4)
        coords = centroids.values.tolist()
                    
        if coords == last_coords:
            break
        
    print(f'Total Iterations: {i + 1}')
    
    data['sqrt_dist_centroid'] = (data[dist].min(axis=1))**2
    inertia = data['sqrt_dist_centroid'].sum()
    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.scatterplot(data=data, x = variables[0], y = variables[1], hue = 'cluster', palette = 'tab10', s=50)
        ax = sns.scatterplot(data=centroids, x = variables[0], y = variables[1], s=100, color='black')
        plt.tight_layout()
        plt.show()
    return data['cluster'].astype(int), inertia

clusters, inertia = kmeans(customers, k=2)

inertias = []

for i in range(1,11):
    inertias.append(kmeans(df=customers, k=i)[1])
    
print(inertias)

##### inertia plot #############
fig, ax = plt.subplots()
ax = plt.plot(range(1,11),inertias, marker ='o')
plt.show()


################## with k = 5 ###############
cols_to_keep = ['Annual Income', 'Spending Score']
customers = customers[cols_to_keep].copy()

clusters, inertia = kmeans(df = customers, k=5, plot=True)