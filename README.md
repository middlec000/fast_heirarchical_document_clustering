# Overview
This project provides a Fast Hierarchical (text) Document Clustering (FHDC) Python package. This algorithm implements agglomerative hierarchical document clustering based on document topic. The original text documents are converted into TF-IDF vectors. At the start of clustering each document is a separate cluster and at each step the two least different (most similar) clusters are combined.  

## Target Use-Case
This algorithm is intended for use on large corpuses where resources are a concern. The goal is to produce useful clusters while minimizing memory and compute resources.

## Summary of Efficiency Boosting Methods
Preprocessing
* Low frequency words are removed
* Document vectors are stored as dictionaries with only present words
 
Clustering  
* Only original documents and current clusters are stored - intermediate steps are recorded as small ClusterNode objects
* Efficient distance metric
* Cluster metrics are additive

Products
* Lazy computing of cluster themes

## History
This project is based on a class project for CSCD 530 Big Data Analytics at Eastern Washington University in winter quarter of 2021.  

That project was a collaboration with Will Hall and the repo can be found [here](https://github.com/use/cord19clustering). Will developed and implemented many of the efficiency-boosting practices for the original K-Means version of the project.  

I took that class project and developed it into its current state.

# Example Output
Small Example  
* [Text](results/example_output/small_example.txt)
* [Dendrogram](results/example_output/small_example_dendrogram.png)  
 
Data Science Cover Letters  
* [Text](results/example_output/cover_letter.txt)
* [Dendrogram](results/example_output/cover_letter_dendrogram.png)

# Preprocessing
The original format for the corpus, or entire collection of documents, is a dictionary with keys of document names and values of the documents texts. The process of preprocessing the documents follows the steps in the diagram below.

## Term Frequency-Inverse Document Frequency (TF-IDF)
According to [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idfhttps://en.wikipedia.org/wiki/Tf%E2%80%93idf) there are several variations of TF-IDF.   

I chose a custom simplified version for efficiency of initial computation and ease of merging clusters - this metric is additive between clusters. This metric still produces human intelligible clusters and cluster themes.  

For a given term $t$ in document $d$, I define TF-IDF as the following:
$$
\text{TF-IDF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Number of times term } t \text{ appears in the corpus overall}}
$$

# Clustering
## Distance Metrics
### Default
### Shipped
### User Defined
## Agglomerative