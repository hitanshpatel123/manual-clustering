import pandas as pd


######################################### working towards final function ##########################
cols_to_keep = ['Annual Income', 'Spending Score']

customers = pd.read_csv('mall_customers.csv')
customers = customers[cols_to_keep]

def get_centroids(data, k):
    cent = data.sample(k).reset_index().drop('index', axis =1)
    coor = cent.values.tolist()
        
    return cent, coor

centroids, coords = get_centroids(customers, k=2)

def calculate_distance(data, coor):
    dist_names = []
    for i, centroid in enumerate(coor):
        name = f'dist_centroid_{i + 1}'
        data[name] = np.sqrt((data.iloc[:, 0] - centroid[0])**2 + (data.iloc[:, 1] - centroid[1])**2)
        dist_names.append(name)
    return data, dist_names

customers , dist_names = calculate_distance(customers, coords)

customers['cluster'] = customers[dist_names].idxmin(axis=1)
# customers['cluster'] = customers['cluster'].str[-1]
customers['cluster'] = customers['cluster'].str.split('_').str[-1]

figure, ax = plt.subplots()
ax = sns.scatterplot(data = customers, y ='Spending Score', x = 'Annual Income', hue = 'cluster', )
ax = sns.scatterplot(data = centroids, y ='Spending Score', x = 'Annual Income', s = 100, color = 'black')
plt.show()

variables = customers.columns[:2]
new_centroids = customers[variables].groupby(customers['cluster']).mean().reset_index().drop('cluster', axis = 1).round(4)

new_coords = new_centroids.values.tolist()

for i in range(100): 
    last_coords = coords.copy()

    customers, dist_names = calculate_distance(customers, coords)

    customers['cluster'] = customers[dist_names].idxmin(axis=1).str.split('_').str[-1]

    centroids = round(customers.groupby('cluster')[variables].mean(), 4)
    coords = centroids.values.tolist()
                    
    if coords == last_coords:
        break

print(f'Total Iterations: {i + 1}')
                    
sns.scatterplot(x='Annual Income', y='Spending Score', hue='cluster', palette='tab10', data=customers, s=50)
centroids.plot.scatter(x='Annual Income', y='Spending Score', color='black', s=100)

plt.show()

###########################################################################################################################

customers = pd.read_csv('mall_customers.csv')

cols_to_keep = ['Annual Income', 'Spending Score']
customers = customers[cols_to_keep].copy()

def kmeans(data, k, n_iterations = 100):
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.scatterplot(data=data, x = variables[0], y = variables[1], hue = 'cluster', palette = 'tab10', s=50)
    ax = sns.scatterplot(data=centroids, x = variables[0], y = variables[1], s=100, color='black')
    plt.tight_layout()
    plt.show()
    return data['cluster']

clusters = kmeans(customers, k=2)