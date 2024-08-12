import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


class FeatureComparison:
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def cosine_similarity_matrix(self, target_feature_vector, feature_vectors):
        return cosine_similarity([target_feature_vector], feature_vectors)[0]

    def get_top_similar(self, k, similarities):
        top_k = {}
        for i, similarity in similarities:
            if similarity >= self.threshold and k > 0:
                top_k[i] = similarity
                k -= 1

        return sorted(top_k, key=top_k.get, reverse=True)
