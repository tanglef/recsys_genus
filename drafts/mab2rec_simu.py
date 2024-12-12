# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import random
from sklearn.metrics import confusion_matrix

seed = 42
np.random.seed(seed)


def simulate_classification_scenario(n=3000, d=10, num_classes=8, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate synthetic data
    X, y = make_classification(
        n_samples=n,
        n_features=d,
        n_classes=num_classes,
        n_informative=d,
        n_redundant=0,
        random_state=seed,
    )

    # Train classifiers on specific subsets of data
    classifiers = []
    class_map = {
        1: [1, 2],  # Classifier 1 - Classes 1, 2
        2: [3, 4, 5, 6],  # Classifier 2 - Classes 3, 4, 5, 6
        3: [1, 3, 6, 7, 8],  # Classifier 3 - Classes 1, 3, 6, 7, 8
        4: [6, 7],  # Classifier 4 - Classes 2, 3
        5: [1, 5, 7, 8],  # Classifier 5 - Classes 5, 7
    }

    X_subset, X_rec, y_subset, y_rec = train_test_split(
        X, y, train_size=int(n / 4), random_state=seed
    )
    clf_feat = []
    for clf_id, classes in class_map.items():

        # Train a classifier (Random Forest for this example)
        clf = RandomForestClassifier(
            random_state=seed + clf_id,
            max_depth=d - clf_id,
        )
        clf.fit(X_subset, y_subset)
        classifiers.append((clf, classes))
        clf_feat.append(np.diag(confusion_matrix(y_subset, clf.predict(X_subset))))
    # Log DataFrame initialization
    log_data = {
        "user_id": [],
        "item_id": [],
        "rating": [],
        "prediction": [],
        "prediction_probability": [],
    }

    nt = X_subset.shape[0]
    # Simulate sequential prediction for 500 new data points
    for i in range(n - nt):
        # Generate new data point
        item_id = i
        X_new, y_true = X_rec[i], y_rec[i]

        num_classifiers = random.randint(2, 5)
        selected_classifiers = random.sample(classifiers, num_classifiers)
        for user_id, (clf, classes) in enumerate(selected_classifiers, start=1):
            if y_true + 1 in classes:
                # Classifier can predict for this class
                probas = clf.predict_proba(X_new.reshape(1, -1))[0]
                prediction = np.argmax(probas)
                prediction_probability = probas[prediction]

                # Log entry if classifier is applicable
                log_data["user_id"].append(user_id)
                log_data["item_id"].append(item_id)
                log_data["rating"].append(1)
                log_data["prediction"].append(prediction)
                log_data["prediction_probability"].append(prediction_probability)
            else:
                # Classifier does not make a prediction for this class
                log_data["user_id"].append(user_id)
                log_data["item_id"].append(item_id)
                log_data["rating"].append(0)
                log_data["prediction"].append(None)
                log_data["prediction_probability"].append(None)

    # Convert log data to DataFrame
    log_df = pd.DataFrame(log_data)
    return log_df, X_rec, y_rec, clf_feat, class_map


#
df, X_rec, y_rec, clf_feat, class_map = simulate_classification_scenario()
print(df.head())


# %%
def stratified_split(data, test_size=0.2):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Perform stratified split by user
    for user_id, user_data in data.groupby("user_id"):
        user_train, user_test = train_test_split(
            user_data, test_size=test_size, random_state=42
        )
        train_data = pd.concat([train_data, user_train])
        test_data = pd.concat([test_data, user_test])

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


train_data, test_data = stratified_split(df)
print("Train Data:\n", train_data)
print("Test Data:\n", test_data)
train_data.columns = [
    "user_id",
    "item_id",
    "response",
    "prediction",
    "prediction_probability",
]
test_data.columns = [
    "user_id",
    "item_id",
    "response",
    "prediction",
    "prediction_probability",
]

train_data.to_csv("./train_data.csv", index=False)
test_data.to_csv("./test_data.csv", index=False)
X_rec = pd.DataFrame(X_rec)
X_rec["item_id"] = np.arange(X_rec.shape[0])
X_rec.to_csv("./features_item.csv", index=False)


# %%
def mapping_to_dataframe(user_class_mapping):
    all_classes = sorted(
        set(cls for classes in user_class_mapping.values() for cls in classes)
    )
    df = pd.DataFrame(columns=["user_id"] + all_classes)
    data = []
    for user_id, classes in user_class_mapping.items():
        row = [user_id] + [1 if cls in classes else 0 for cls in all_classes]
        data.append(row)
    df = pd.DataFrame(data, columns=["user_id"] + all_classes)
    return df


user_features = mapping_to_dataframe(class_map)
user_features.to_csv("./features_user.csv", index=False)
# %%

from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics

# Column names for the response, user, and item id columns
metric_params = {
    "click_column": "score",
    "user_id_column": "user_id",
    "item_id_column": "item_id",
}

# Evaluate peformance at different k-recommendations
top_k_list = [2, 5, 10]

# List of metrics to benchmark
metrics = []
for k in top_k_list:
    metrics.append(BinaryRecoMetrics.AUC(**metric_params, k=k))
    metrics.append(BinaryRecoMetrics.CTR(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Precision(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Recall(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.NDCG(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.MAP(**metric_params, k=k))

metrics

# %%
from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy

# Set of recommenders
recommenders = {
    "Random": BanditRecommender(learning_policy=LearningPolicy.Random(), seed=seed),
    "LinGreedy": BanditRecommender(
        learning_policy=LearningPolicy.LinGreedy(epsilon=0.1), seed=seed
    ),
    "LinTS": BanditRecommender(learning_policy=LearningPolicy.LinTS(), seed=seed),
    "ClustersTS": BanditRecommender(
        learning_policy=LearningPolicy.ThompsonSampling(),
        neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=3),
        seed=seed,
    ),
}

# %%
from mab2rec.pipeline import benchmark

# Benchmark the set of recommenders for the list of metrics
# using training data and user features scored on test data
reco_to_results, reco_to_metrics = benchmark(
    recommenders,
    metrics=metrics,
    train_data="./train_data.csv",
    test_data="./test_data.csv",
    user_features="./features_user.csv",
    item_features="./features_item.csv",
)

# %%
from mab2rec.visualization import plot_metrics_at_k

# Plot each metric across all algorithms at different k
plot_metrics_at_k(reco_to_metrics, col_wrap=2)

# %%
from mab2rec.visualization import plot_inter_diversity_at_k

# Plot diversity at each k for each algorithm
plot_inter_diversity_at_k(reco_to_results, k_list=top_k_list, col_wrap=2)

# %%
from mab2rec.visualization import plot_recommended_counts

# Plot recommendation counts (average=False)
plot_recommended_counts(
    reco_to_results, test_data, k=5, alpha=0.7, average_response=False, col_wrap=2
)

# %%
from mab2rec.visualization import plot_recommended_counts_by_item

# Plot recommendation counts (normalize=True)
plot_recommended_counts_by_item(
    reco_to_results, k=5, top_n_items=15, normalize=True, col_wrap=2
)

# %%
from mab2rec.visualization import plot_intra_diversity_at_k

plot_intra_diversity_at_k(reco_to_results, X_rec, k_list=[2, 5, 10], col_wrap=2)

# %%
reclin = reco_to_results["LinTS"]
reclin["class"] = reclin["item_id"].apply(lambda item_id: y_rec[item_id])

# %%
