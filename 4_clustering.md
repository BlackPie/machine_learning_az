# Part 4: Clustering

**Clustering** is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

## Section 21: K-Means Clustering
![image](images/33.png)
That’s how it works:
* Step 1. Choose the number K of clusters 
* Step 2. Select random points which will be controids of clusters (not necessarily from the dataset)
* Step 3. Assign each data point to the closest centroid. That forms K clusters
* Step 4. Compute and place new controids for all clusters
* Step 5. If there are no reassignments, the model is done. Otherwise, repeat steps 3 and 4.

The algorythm has a tricky thing you should be aware of - random initialization trap. Sometimes random centroids can be placed in unlucke places which can lead to not quite accureate results
Expected result: ![image](images/34.png)
Actual result: ![image](images/35.png)
That’s why there is updated model K-Means++. Description of this model and the way it choose initial centroids is out of the scope of the course, but what we need to know is that all libraries (sklearn in our case) already use it.

Choosing the right number of clusters
WCSS stands for … and it is sum of squared sums of distances between all points and a centroid for every cluster. The more clusters you have the smaller the value will become. You have to create a chart and find optimal numer of clasters. It is called The Elbow Method: ![image](images/36.png)

```
# TODO: put kmeans_python here
```

## Section 22: Hierarchical Clustering

There are two types of HIerarchical Clustering: Agglomerative and Divisive.
Agglomerative HC:
* Step 1. Make each data point a single-point cluster. That forms N clusters.
* Step 2. Take the two closest data points(clusters) and make them a single cluster. That forms N-1 clusters
closest clusters are 
* Step 3. Repeat step 2 until there is only one cluster left

Divisive HC is the same thing but with reversed steps.

How Dendrograms Work.
Dendogram is kind of memory for a HC algorythm. It remembers every step of the algorythm. It contains distances between two merged clusters for every step. The bigger distance, the bigger dissimilarity of features of clusters: ![image](images/37.png)

There is no the only correct way to use dendrograms.
We can set maximum level of dissimilarity, which will decline N last merges and leave multiple clusters as the result: ![image](images/38.png)
Standard approach is to look for the highest vertical distance which doesn’t cross extended horizontal lines and revoke related merges: ![image](images/39.png)
Not sure if it is correct :(
`# TODO: check other sources to find information how to use dendrograms`
Here is example of a largest distance which doesn’t cross any horizontal line: ![image](images/40.png)

Taken from the internet:
```
To determine the cutting section, various methods can be used. The first method, empirically, this method should be based on knowledge of researcher. For example, the existing cluster based on appearance differences, can be divided into three or four groups.

The second method uses a statistical conventions. The dendrogram can be cut where the difference is most significant. Another technique is to use the square root of the number of individuals. Another technique is to use at least 70% of the distance between the two groups. The next method is to use the function discrimination and classification based on discrimination function.
```

Implementation
```
# TODO: put the HC template here
```

Pros and cons of two clustering algorythms: [link](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Clustering-Pros-Cons.pdf)



