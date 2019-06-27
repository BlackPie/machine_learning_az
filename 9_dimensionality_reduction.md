 Part 9: Dimensionality Reduction

Section 34
————————————
Principal Component Analysis

PCA is one of the most used unsupervised algorithms and most popular dimensionality reduction algorithm. It can be used for noise filtering, visualizetion, feature extraction, stock market predictions or gene data analysis. It detects the correlation between variables and identifies patterns in data. The main goal of it is to reduce dimensions of d-dimension dataset by projecting it onto k-dimensional subspace where k < d

Another explanation:
From m independent variables of your dataset, PCA extracts p < m independent variables that explain the most the variance of the dataset, regardles of the dependent variable.

```
# TODO: put the code here
```

It gives us extra information to decide how many variables we should use
It shows how much every variable explains the variance.
```
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance_vector = pca.explained_variance_ratio_
```

Section 35
————————————
Linear Discriminant Analysis

LDA
Used as dimensionality reduction techique
Used in the pre processing step for pattern classification
has the goal to project a dataset onto a lower dimensional space
Sounds similar to PCA but in addition to finding the component axises with LDA we are interested in the axes that maximize the separation between multiple classes. 

The goal of LDA is to project a feature space onto a small subspace while maintaining the class-discriminatory information. Both PCA and LDA are linear transformation techniques used for dimensional reduction. PCA is described as unsupervised but LDA is supervised because of the relation to the dependent variable.
http://prntscr.com/o6udjn

LDA extracts news independent variables from old independent variables that separate the most the classes of the dependent variables 

```
PUT THE CODE
```

Section 36
————————————
Kernel PCA

Basically it’s just a version of PCA which can be applied for non linear problems.
It creates new features that way so a non linear problem becomes more separable.
Before: http://prntscr.com/o6vw7f
After: http://prntscr.com/o6vxg7
Or this:  http://prntscr.com/o6w07n



