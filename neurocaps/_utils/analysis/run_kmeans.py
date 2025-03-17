"""Internal function for performing cluster selection with or without joblib"""

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score


def _run_kmeans(n_cluster, configs, concatenated_timeseries, method):
    model = KMeans(n_clusters=n_cluster, **configs, verbose=0).fit(concatenated_timeseries)

    # Only return model when no cluster selection chosen
    if method is None:
        return model

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

    model_dict = {n_cluster: model}

    return performance, model_dict
