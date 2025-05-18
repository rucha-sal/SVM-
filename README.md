Importing necessary libraries:
 - `pandas` for data manipulation and analysis.
 - `sklearn.datasets.load_wine` to load the Wine dataset.
 - `matplotlib.pyplot` for creating visualizations.
 - `numpy` for numerical operations.
 - `sklearn.svm` for support vector machine classification.
 - `sklearn.metrics` for evaluating the model.
 - `seaborn` for additional visualization options.

Loading the Wine dataset:
 - The code uses `load_wine()` from `sklearn.datasets` to load the Wine dataset.
Exploring the attributes of the dataset:
 - The `dir(wine)` function is used to list the attributes and methods of the dataset.

Preparing the data for training and testing:
 - `X` is assigned the feature data (attributes) from the dataset.
 - `y` is assigned the target variable (class labels)

Splitting the data into training and testing sets:
 - `train_test_split()` is used to split the dataset into training and testing sets with 80% for
training and 20% for testing.
Training a support vector machine (SVM) model:
 - The SVM classifier (`SVC`) is used to train the model on the training data.

Making predictions and calculating accuracy:
 - `y_pred` holds the predicted labels for the test set using the trained SVM model.
 - `accuracy_score` is used to compute the accuracy of the model on the test set (`y_test`).

Confusion matrix and visualization:
Confusion matrix is calculated using `confusion_matrix` to evaluate the model's
performance.
The confusion matrix is visualized using `seaborn` and `matplotlib` to display true labels
versus predicted labels.
`X.shape` is used to determine the shape of the features array (`X`). The `shape` attribute
of a NumPy array returns a tuple representing the dimensions of the array.
For the Wine dataset, `X.shape` gives the dimensions in the form `(n_samples, n_features)`,
where:
`n_samples` is the number of samples (rows) in the dataset.
`n_features` is the number of features (columns) for each sample.
 - `y.shape` is used to determine the shape of the target variable array (`y`). Similar to
`X.shape`, this gives the dimensions of the array in the form `(n_samples,)`, where
`n_samples` is the number of samples (rows) in the dataset, representing the number of
labels corresponding to the samples.
