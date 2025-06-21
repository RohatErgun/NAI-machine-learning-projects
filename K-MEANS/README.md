
### K-Means Clustering (Iris Dataset)

**This project implements the K-Means clustering algorithm in Python using the Iris dataset.**

Files
kmeans.py: Python script that contains the implementation of the K-Means algorithm.
iris.data: Dataset file containing the Iris data (4 features per instance, no labels).

#### Install dependencies

```bash
pip install numpy
```

#### How It Works

1. **Data Loading**: The program reads the Iris dataset from a file and extracts the first four columns as feature vectors.  
2. **Normalization**: Features are standardized to have zero mean and unit variance.  
3. **K-Means Steps**:
   - Randomly initialize `k` centroids.
   - Assign each data point to the nearest centroid.
   - Update centroids based on the mean of the assigned points.
   - Repeat until convergence or a fixed number of iterations.

#### How to Run

Run the script using:

```bash
python kmeans.py
```

Make sure you initialize and run the `Kmeans` class in your script like this:

```python
model = Kmeans(k=3, filename="iris.data")
model.run()
```
You can change the value of `k` to set a different number of clusters.
