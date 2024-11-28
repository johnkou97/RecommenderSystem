# Recommender Systems - MovieLens 1M Dataset

## Introduction

This project was part of the course "Data Mining" at Leiden University. We implemented several recommendation algorithms using the MovieLens 1M dataset, which contains 1,000,209 ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000. The ratings are on a 5-star scale (whole-star ratings only). Each user has at least 20 ratings. The dataset also includes user information about their gender, age, occupation and zip-code, and movie information about the title and genres. The whole dataset can be downloaded from [here](https://grouplens.org/datasets/movielens/1m/).

## Files

- `ml-1m/`: Folder containing the MovieLens 1M dataset.
- `Exploratory.ipynb`: Jupyter notebook with the exploratory data analysis of the dataset.
- `Recommender.ipynb`: Jupyter notebook with the implementation of the recommendation algorithms.
- `plots/`: Folder containing the plots generated during the analysis.
- `M_1.npy` & `U_1.npy`: Files containing the learned matrices for the UV decomposition model. These files are generated during the training of the model and are used to make predictions. They are saved in the root directory to be used in the PCA, t-SNE, and UMAP plots.
- `report.pdf`: final report of the project, including the results and the analysis.
- `README.md`: This file.

## Running the Code

First make sure you have the required libraries installed and the dataset downloaded (the dataset is already included in the repository). You can then run the Jupyter notebooks `Recommender.ipynb` for the implementation of the recommendation algorithms and `Exploratory.ipynb` for the exploratory data analysis, where you can see the PCA, t-SNE, and UMAP plots.

## Results

We used the following algorithms:
- Naive Approaches: global average rating, average rating per item, average rating per user, and an "optimal" linear combination of the two averages (per user and per item), with and without the parameter γ.
- UV matrix decomposition algorithm
- Matrix Factorization with Gradient Descent and Regularization

For the validation, we used a 5-fold cross-validation, where the dataset was randomly divided into 5 parts, and each part was used as a test set once while the other parts were used as training sets. The RMSE and MAE were calculated for each fold and averaged over the 5 folds. This is a common technique to evaluate the performance of a model and make sure it generalizes well to unseen data.

During training, we kept track of the execution time and memory usage of each model, as well as the RMSE and MAE on each epoch. The plots below show the RMSE and MAE of the UV decomposition and Matrix Factorization models during training.

![UV Decomposition](/plots/uv.png)<br />
*RMSE (solid line) and MAE (dashed line) during training for both the training (blue) and validation (purple) sets for the UV Decomposition model.*

![Matrix Factorization Zoom](/plots/matrixfactorization_errors.png)<br />
*Same as above, but for the Matrix Factorization model.*



The final comparison of the models shows that the Matrix Factorization model performed the best, with the lowest RMSE and MAE on the test set. The UV Decomposition model was not better than some of the naive approaches, which makes it a poor choice for this dataset. The final performance of the models, with the execution time and memory usage, is shown in the plots below.

![Execution Time](/plots/comparison_time.png)<br />
*The RMSE (solid line) and MAE (dashed line) of the final model on the training (blue) and validation (purple) sets. With the red line indicating the execution time.*

![Memory Usage](/plots/comparison_memory.png)<br />
*Same as above, but with the red line indicating the memory usage.*


The results can, also, be summarized in a table:

| Model | RMSE Test | MAE Test | Execution Time (s) | Memory (MB) |
| --- | --- | --- | --- | --- |
| Global Average | 1.117 | 0.934 | 15.330 | 375.5 |
| Movie Average | 0.980 | 0.782 | 88.707 | 389.4 |
| User Average | 1.035 | 0.829 | 55.218 | 389.4 |
| Lin Reg with intercept | 0.924 | 0.733 | 178.778 | 416.6 |
| Lin Reg without intercept | 0.953 | 0.763 | 172.274 | 447.5 |
| UV Decomposition | 0.871 | 0.828 | 822.069 | 1948.0 |
| Matrix factorisation | 0.885 | 0.689 | 1190.862 | 2054.6 |

<p></p>

The next part of the assignment was to use dimensionality reduction techniques to visualize the data. We used three different techniques:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

We first used the dimensionality reduction techniques to cluster the users based on their gender and see if the clusters were separable. The plots below show the results of the three techniques, where the colors represent the gender of the users.

![PCA](/plots/pca/gender.png)<br />
*Clustering of the data using PCA. The colors represent the different genders of the users.*

![tSNE](/plots/tsne/gender.png)<br />
*Same as above, but using tSNE.*

![UMAP](/plots/umap/gender.png)<br />
*Same as above, but using UMAP.*

As it is clear, the different genders are not separable using these techniques. We can also see that the PCA plots are not very informative, as all the points are clustered together. We created many more plots using the same techniques and different features. On the plots below, we clustered the movies based on 2 genres (horror and romantic) using t-SNE and UMAP. In this case we can see that the clusters are more separable, especially using UMAP.

![tSNE](/plots/tsne/genre.png) <br />
*Clustering of the data using tSNE. The colors represent two different genres of the movies, horror (blue) and romantic (orange).*

![UMAP](/plots/umap/genre.png) <br />
*Same as above, but using UMAP.*


