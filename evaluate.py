from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#Different methods for finding and comparing clustering. For purity we need true labels which were not there, this used some other methods for result analysis
def evaluate_silhouette(features, labels):
    return silhouette_score(features, labels)

def evaluate_calinski_harabasz(features, labels):
    return calinski_harabasz_score(features, labels)

def evaluate_davies_bouldin(features, labels):
    return davies_bouldin_score(features, labels)


# Combine all metrics for evaluation
def evaluate_clustering(features, labels, n_clusters=5):
    silhouette = evaluate_silhouette(features, labels)
    calinski_harabasz = evaluate_calinski_harabasz(features, labels)
    davies_bouldin = evaluate_davies_bouldin(features, labels)
    
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    
    return {
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Davies-Bouldin Index": davies_bouldin,
    }
