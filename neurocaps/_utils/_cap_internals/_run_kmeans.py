"""Internal function for performing silhouette or elbow method with or without multiprocessing"""
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

def _run_kmeans(n_cluster, random_state, init, n_init, max_iter, tol, algorithm, concatenated_timeseries, method):
    model = KMeans(n_clusters=n_cluster, random_state=random_state, init=init, n_init=n_init, max_iter=max_iter,
                   tol=tol, algorithm=algorithm).fit(concatenated_timeseries)

    cluster_labels = model.labels_

    if method == "davies_bouldin":
        performance = {n_cluster: davies_bouldin_score(concatenated_timeseries, cluster_labels)}
    elif method == "elbow":
        performance = {n_cluster: model.inertia_}
    elif method == "silhouette":
        performance = {n_cluster: silhouette_score(concatenated_timeseries, cluster_labels, metric="euclidean")}
    else:
        # Variance Ratio
        performance = {n_cluster: calinski_harabasz_score(concatenated_timeseries, cluster_labels)}

    return performance
