# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd
import numpy as np
from scipy.io import loadmat

# ----------------------------------- Create complex network --------------------------------
input_path = "D:\\industry_chain\\1Attribute_analysis\\"
z_matrix_path = input_path + "Z_single_3industries.mat" # USD billion
sector_info_path = input_path + "3industry_information.xlsx"
region_info_path = input_path + "region_information.xlsx"

file = loadmat(z_matrix_path)
z_matrix = file['Z_single'][:] 
sectors = pd.read_excel(sector_info_path, usecols=['Sector_Chinese_name'])
regions = pd.read_excel(region_info_path, usecols=['City_Chinese_name'])


# create graph
G = nx.DiGraph()

# Add note
for region in regions['City_Chinese_name']:
    for sector in sectors['Sector_Chinese_name']:
        G.add_node(f"{region}_{sector}")

# Add edge and weight
for i, region1 in enumerate(regions['City_Chinese_name']):
    for j, region2 in enumerate(regions['City_Chinese_name']):
        for k, sector1 in enumerate(sectors['Sector_Chinese_name']):
            for l, sector2 in enumerate(sectors['Sector_Chinese_name']):
                weight = z_matrix[i*3 + k, j*3 + l]
                if weight > 0:
                    G.add_edge(f"{region1}_{sector1}", f"{region2}_{sector2}", weight=weight)

# ----------------------------------- Calculate betweenness --------------------------------
import networkx as nx
H1 = nx.DiGraph()
for u, v, data in G.edges(data=True):
    weight = data['weight']  

    if weight <= 0:
        continue

    H1.add_edge(u, v, weight=1 / weight)


edges_with_weights = {}

for u, v, data in H1.edges(data=True):
    edges_with_weights[(u, v)] = data['weight']
max_weight = max(edges_with_weights.values())
min_weight = min(edges_with_weights.values())


betweenness = nx.betweenness_centrality(H1, normalized=True, weight='weight')

max_betweenness = max(betweenness.values())
normalized_betweenness = {node: value / max_betweenness for node, value in betweenness.items()}

min_normalized_betweenness = min(normalized_betweenness.values())
max_normalized_betweenness = max(normalized_betweenness.values())

min_betweenness = min(betweenness.values())
max_betweenness = max(betweenness.values())


city_betweenness = {}
industry_betweenness = {}


import matplotlib.pyplot as plt

city_sum_and_count = {}
industry_sum_and_count = {}

for node, value in normalized_betweenness.items():
    city, industry = node.split('_')
    city_sum_and_count[city] = city_sum_and_count.get(city, (0, 0))
    industry_sum_and_count[industry] = industry_sum_and_count.get(industry, (0, 0))

    city_total, city_count = city_sum_and_count[city]
    industry_total, industry_count = industry_sum_and_count[industry]

    city_sum_and_count[city] = (city_total + value, city_count + 1)
    industry_sum_and_count[industry] = (industry_total + value, industry_count + 1)


avg_city_normalized_betweenness = {city: total / count for city, (total, count) in city_sum_and_count.items()}
avg_industry_normalized_betweenness = {industry: total / count for industry, (total, count) in industry_sum_and_count.items()}


sorted_avg_city_normalized_betweenness = dict(sorted(avg_city_normalized_betweenness.items(), key=lambda item: item[1], reverse=True))

N = 10  
top_cities = dict(list(sorted_avg_city_normalized_betweenness.items())[:N])


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(top_cities.keys(), top_cities.values(), color='blue')
plt.title('Average Normalized Betweenness Centrality by City')
plt.xlabel('City')
plt.ylabel('Average Normalized Betweenness Centrality')

plt.subplot(1, 2, 2)
plt.bar(avg_industry_normalized_betweenness.keys(), avg_industry_normalized_betweenness.values(), color='green')
plt.title('Average Normalized Betweenness Centrality by Industry')
plt.xlabel('Industry')
plt.ylabel('Average Normalized Betweenness Centrality')

plt.tight_layout()
plt.show()

# ----------------------------------- Calculate closeness --------------------------------
H2 = nx.DiGraph()
max_original_weight = max(data['weight'] for u, v, data in G.edges(data=True))
for u, v, data in G.edges(data=True):
    weight = data['weight']
    if weight <= 0:
        continue
    H2.add_edge(u, v, weight=1 / weight)

edges_with_weights = {}

for u, v, data in H2.edges(data=True):
    edges_with_weights[(u, v)] = data['weight']
max_weight = max(edges_with_weights.values())
min_weight = min(edges_with_weights.values())


closeness = nx.closeness_centrality(H2,distance='weight') 
max_closeness = max(closeness.values())
normalized_closeness = {node: value / max_closeness for node, value in closeness.items()}

min_normalized_closeness = min(normalized_closeness.values())
max_normalized_closeness = max(normalized_closeness.values())

import matplotlib.pyplot as plt

city_sum_and_count = {}
industry_sum_and_count = {}

for node, value in normalized_closeness.items():
    city, industry = node.split('_')
    city_sum_and_count[city] = city_sum_and_count.get(city, (0, 0))
    industry_sum_and_count[industry] = industry_sum_and_count.get(industry, (0, 0))

    city_total, city_count = city_sum_and_count[city]
    industry_total, industry_count = industry_sum_and_count[industry]

    city_sum_and_count[city] = (city_total + value, city_count + 1)
    industry_sum_and_count[industry] = (industry_total + value, industry_count + 1)


avg_city_normalized_closeness = {city: total / count for city, (total, count) in city_sum_and_count.items()}
avg_industry_normalized_closeness = {industry: total / count for industry, (total, count) in industry_sum_and_count.items()}


sorted_avg_city_normalized_closeness = dict(sorted(avg_city_normalized_closeness.items(), key=lambda item: item[1], reverse=True))

N = 10  
top_cities = dict(list(sorted_avg_city_normalized_closeness.items())[:N])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(top_cities.keys(), top_cities.values(), color='blue')
plt.title('Average Normalized Closeness Centrality by City')
plt.xlabel('City')
plt.ylabel('Average Normalized Closeness Centrality')

plt.subplot(1, 2, 2)
plt.bar(avg_city_normalized_closeness.keys(), avg_city_normalized_closeness.values(), color='green')
plt.title('Average Normalized Closeness Centralityby Industry')
plt.xlabel('Industry')
plt.ylabel('Average Normalized Closeness Centrality')

plt.tight_layout()
plt.show()
# ----------------------------------- Calculate degree centrality --------------------------------

def weighted_degree_centrality(G):

    if G.number_of_nodes() <= 1:
        return {n: 0 for n in G}

    max_degree = (G.number_of_nodes() - 1) * max([d.get('weight', 1) for u, v, d in G.edges(data=True)])
    centrality = {}  
    for n in G:
        in_degree = sum([d.get('weight', 1) for u, v, d in G.in_edges(n, data=True)])
        out_degree = sum([d.get('weight', 1) for u, v, d in G.out_edges(n, data=True)])
        weighted_degree = in_degree + out_degree
        centrality[n] = weighted_degree / max_degree

    return centrality


degree_centrality = weighted_degree_centrality(G)

min_degree_centrality = min(degree_centrality.values())
max_degree_centrality = max(degree_centrality.values())

import networkx as nx
import matplotlib.pyplot as plt

city_sum_and_count = {}
industry_sum_and_count = {}

for node, value in degree_centrality.items():
    city, industry = node.split('_')
    city_sum_and_count[city] = city_sum_and_count.get(city, (0, 0))
    industry_sum_and_count[industry] = industry_sum_and_count.get(industry, (0, 0))

    city_total, city_count = city_sum_and_count[city]
    industry_total, industry_count = industry_sum_and_count[industry]

    city_sum_and_count[city] = (city_total + value, city_count + 1)
    industry_sum_and_count[industry] = (industry_total + value, industry_count + 1)

avg_city_degree_centrality = {city: total / count for city, (total, count) in city_sum_and_count.items()}
avg_industry_degree_centrality = {industry: total / count for industry, (total, count) in industry_sum_and_count.items()}


sorted_avg_city_degree_centrality = dict(sorted(avg_city_degree_centrality.items(), key=lambda item: item[1], reverse=True))

N = 10  
top_cities = dict(list(sorted_avg_city_degree_centrality.items())[:N])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(top_cities.keys(), top_cities.values(), color='blue')
plt.title('Average degree_centrality by City')
plt.xlabel('City')
plt.ylabel('Average degree_centrality')

plt.subplot(1, 2, 2)
plt.bar(avg_industry_degree_centrality.keys(), avg_industry_degree_centrality.values(), color='green')
plt.title('Average degree_centrality by Industry')
plt.xlabel('Industry')
plt.ylabel('Average degree_centrality Centrality')

plt.tight_layout()
plt.show()

# ----------------------------------- Calculate eigenvector centrality--------------------------------
import networkx as nx

eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
min_eigenvector_centrality = min(eigenvector_centrality.values())
max_eigenvector_centrality = max(eigenvector_centrality.values())

import networkx as nx
import matplotlib.pyplot as plt

city_sum_and_count = {}
industry_sum_and_count = {}

for node, value in eigenvector_centrality.items():
    city, industry = node.split('_')
    city_sum_and_count[city] = city_sum_and_count.get(city, (0, 0))
    industry_sum_and_count[industry] = industry_sum_and_count.get(industry, (0, 0))

    city_total, city_count = city_sum_and_count[city]
    industry_total, industry_count = industry_sum_and_count[industry]

    city_sum_and_count[city] = (city_total + value, city_count + 1)
    industry_sum_and_count[industry] = (industry_total + value, industry_count + 1)

avg_city_eigenvector_centrality = {city: total / count for city, (total, count) in city_sum_and_count.items()}
avg_industry_eigenvector_centrality = {industry: total / count for industry, (total, count) in industry_sum_and_count.items()}

sorted_avg_city_eigenvector_centrality = dict(sorted(avg_city_eigenvector_centrality.items(), key=lambda item: item[1], reverse=True))

top_cities = dict(list(sorted_avg_city_eigenvector_centrality.items())[:N])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(top_cities.keys(), top_cities.values(), color='blue')
plt.title('Average eigenvector_centrality by City')
plt.xlabel('City')
plt.ylabel('Average eigenvector_centrality')

plt.subplot(1, 2, 2)
plt.bar(avg_industry_eigenvector_centrality.keys(), avg_industry_eigenvector_centrality.values(), color='green')
plt.title('Average eigenvector_centrality by Industry')
plt.xlabel('Industry')
plt.ylabel('Average eigenvector_centrality Centrality')

plt.tight_layout()
plt.show()