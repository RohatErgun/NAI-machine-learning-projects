import numpy as np
import random


class Kmeans:
    def __init__(self, k, filename):
        self.k = k
        self.filename = filename
        self.data = []

        self.X = self.load_file()
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)
        self.X_norm = (self.X - self.X_mean) / self.X_std

    def load_file(self):
        with (open(self.filename, "r") as file):
            for line in file:
                if line.strip():
                    parts = line.strip().split(",")
                    features = list(map(float, parts[:4]))
                    self.data.append(features)

        return np.array(self.data)

    def init_centroids(self):
        indices = random.sample(range(len(self.X)), self.k)
        return self.X[indices]

    def assign_clusters_to_closest_centroid(self, centroids):
        labels = []
        for point in self.X:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            closest_index = np.argmin(distances)
            labels.append(closest_index)
        return np.array(labels)

    def update_centroids(self, labels):
        return np.array([self.X[labels == i].mean(axis=0) for i in range(self.k)])

    # sum of distance(xi - centroid assign)
    def calculate_total_distance(self, centroids, labels):
        return sum(
            np.linalg.norm(self.X[i] - centroids[labels[i]]) for i in range(len(self.X))
        )

    def kmeans(self):
        centroids = self.init_centroids()
        old_labels = None
        for iteration in range(1, 100):
            # for each data point compute distance to each centroid
            # assign it to the nearest centroid
            labels = self.assign_clusters_to_closest_centroid(centroids)
            total_distance = self.calculate_total_distance(centroids, labels)

            print(f"Iteration:{iteration} TotalDistance:{total_distance:.2f}")

            if old_labels is not None and np.array_equal(old_labels, labels):
                break

            old_labels = labels.copy()
            centroids = self.update_centroids(labels)


if __name__ == "__main__":
    choice = int(input("Enter a k value: "))

    k = Kmeans(choice, "iris.data")
    k.kmeans()

