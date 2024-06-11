"""Internal function for performing silhouette or elbow method with or without multiprocessing"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _run_kmeans(n_cluster, random_state, init, n_init, max_iter, tol, algorithm, concatenated_timeseries, method):
    model = KMeans(n_clusters=n_cluster, random_state=random_state, init=init, n_init=n_init, max_iter=max_iter,
                   tol=tol, algorithm=algorithm).fit(concatenated_timeseries)
    if method == "silhouette":
        cluster_labels = model.labels_
        performance = {n_cluster: silhouette_score(concatenated_timeseries, cluster_labels, metric="euclidean")}
    else:
        performance = {n_cluster: model.inertia_}

    return performance
