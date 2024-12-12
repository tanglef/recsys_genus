# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn import preprocessing

dir_ = Path(__file__).parents[0] / "data" / "recsys_interactions_minimal.parquet"


def preprocess_plantnet_data(
    logs_path,
    min_number_of_reviews=50,
    min_number_of_participations=50,
    balanced_classes=False,
    use_context=True,
):
    print("preparing ratings log")
    logs = pd.read_parquet(logs_path)
    obs_counts = logs["obs_id"].value_counts()
    obs_to_keep = (
        pd.DataFrame(obs_counts)
        .loc[pd.DataFrame(obs_counts)["count"] >= min_number_of_reviews]
        .index
    )
    logs = logs.loc[logs["obs_id"].isin(obs_to_keep)]
    logs["rating"] = logs["interaction_type"].map(
        {"determination": 1, "quality": 0, "noplant": 0, "organ": 0, "malformed": 0}
    )
    logs = (
        logs[["user_id", "obs_id", "rating"]]
        .groupby(["user_id", "obs_id"])
        .sum()
        .reset_index()
    )
    logs.loc[logs["rating"] > 10, "rating"] = 10
    interactions_count = logs["user_id"].value_counts()
    user_to_keep = (
        pd.DataFrame(interactions_count)
        .loc[pd.DataFrame(interactions_count)["count"] >= min_number_of_participations]
        .index
    )
    logs = logs.loc[logs["user_id"].isin(user_to_keep)]

    if balanced_classes is True:
        print("Balancing")
        logs = logs.groupby("obs_id")
        logs = logs.apply(lambda x: x.sample(logs.size().min()).reset_index(drop=True))
    # shuffle rows to deibas order of user ids
    print("Shuffling")
    logs = logs.sample(frac=1)
    # create a 't' column to represent time steps for the bandit to simulate a live learning scenario
    logs["t"] = np.arange(len(logs))
    logs.index = logs["t"]
    print(
        f"""
    Keeping: {logs["obs_id"].nunique()} obs and {logs["user_id"].nunique()} user with a total of {logs.shape} interations
"""
    )
    if use_context:
        obs_ids = logs["obs_id"].unique()
        user_info = (
            logs[["user_id", "user_weight", "user_determination_family"]]
            .groupby(["user_id"])
            .value_counts()
            .reset_index()
        )
        user_info.drop("count", inplace=True, axis=1)
        user_info = (
            user_info.groupby(["user_id", "user_weight"])
            .value_counts()
            .groupby(level=0, group_keys=False)
            .head(2)
            .reset_index()
        )
        user_info["rank"] = user_info.groupby("user_id").cumcount() + 1
        user_info = user_info.pivot(
            index=[j for j in user_info.columns if j != "rank"],
            columns="rank",
            values="user_determination_family",
        ).reset_index()
        user_info.drop("count", inplace=True, axis=1)
        user_info.columns = ["user_id", "user_weight", "family_1", "family_2"]
    else:
        obs_info = None
        user_info = None
    le_user = preprocessing.LabelEncoder()
    le_item = preprocessing.LabelEncoder()
    logs["user_id"] = le_user.fit_transform(logs["user_id"])
    logs["obs_id"] = le_item.fit_transform(logs["obs_id"])
    return logs, obs_info, user_info


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


logs, *_ = preprocess_plantnet_data(
    dir_, min_number_of_reviews=20, min_number_of_participations=100, use_context=False
)
logs.drop("t", inplace=True, axis=1)
logs.columns = ["user_id", "item_id", "response"]

train_data, test_data = stratified_split(logs)
print("Train Data:\n", train_data)
print("Test Data:\n", test_data)

train_data.to_csv("./train_data.csv", index=False)
test_data.to_csv("./test_data.csv", index=False)

# %%
from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics
from mab2rec.pipeline import benchmark

# Set of recommenders
recommenders = {
    "Random": BanditRecommender(learning_policy=LearningPolicy.Random()),
    # "LinGreedy": BanditRecommender(
    #     learning_policy=LearningPolicy.LinGreedy(epsilon=0.1)
    # ),
    # "LinTS": BanditRecommender(learning_policy=LearningPolicy.LinTS()),
    # "ClustersTS": BanditRecommender(
    #     learning_policy=LearningPolicy.ThompsonSampling(),
    #     neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=10),
    # ),
}

recommenders
# %%
# Column names for the response, user, and item id columns
metric_params = {
    "click_column": "score",
    "user_id_column": "user_id",
    "item_id_column": "item_id",
}

# Evaluate peformance at different k-recommendations
top_k_list = [3, 5, 10]

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
reco_to_results, reco_to_metrics = benchmark(
    recommenders,
    metrics=metrics,
    train_data="./train_data.csv",
    test_data="./test_data.csv",
    user_features=None,
)

# %%
from mab2rec.visualization import plot_metrics_at_k

# Plot each metric across all algorithms at different k
plot_metrics_at_k(reco_to_metrics, col_wrap=2)

# %%
