# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def simulate_and_save_recommender_data(
    num_users=100,
    num_items=500,
    num_user_features=5,
    num_item_features=5,
    test_size=0.2,
    sparsity=0.1,
    seed=42,
):
    np.random.seed(seed)
    user_features = np.random.rand(num_users, num_user_features)
    user_feature_df = pd.DataFrame(
        user_features, columns=[f"user_feature_{i+1}" for i in range(num_user_features)]
    )
    user_feature_df["user_id"] = np.arange(num_users)
    item_features = np.random.rand(num_items, num_item_features)
    item_feature_df = pd.DataFrame(
        item_features, columns=[f"item_feature_{i+1}" for i in range(num_item_features)]
    )
    num_interactions = int(num_users * num_items * sparsity)
    user_item_pairs = np.array(
        np.meshgrid(np.arange(num_users), np.arange(num_items))
    ).T.reshape(-1, 2)
    sampled_pairs = user_item_pairs[
        np.random.choice(user_item_pairs.shape[0], num_interactions, replace=False)
    ]
    item_feature_df["item_id"] = np.arange(num_items)
    interactions = []
    for user_id, item_id in sampled_pairs:
        similarity = np.dot(user_features[user_id], item_features[item_id]) / (
            np.linalg.norm(user_features[user_id])
            * np.linalg.norm(item_features[item_id])
        )
        high_threshold = 0.7
        mid_threshold = 0.5
        if similarity > high_threshold:
            response = 1
        elif similarity > mid_threshold:
            response = 1
        else:
            response = 0
        interactions.append((user_id, item_id, response))
    interactions_df = pd.DataFrame(
        interactions, columns=["user_id", "item_id", "response"]
    )
    train_df, test_df = train_test_split(
        interactions_df, test_size=test_size, random_state=seed
    )
    train_df.to_csv(
        "./simulated_data/train_interactions.csv",
        index=False,
        columns=["user_id", "item_id", "response"],
    )
    test_df.to_csv(
        "./simulated_data/test_interactions.csv",
        index=False,
        columns=["user_id", "item_id", "response"],
    )
    user_feature_df.to_csv(
        "./simulated_data/user_features.csv",
        index=False,
        columns=["user_id"] + [f"user_feature_{i+1}" for i in range(num_user_features)],
    )
    item_feature_df.to_csv(
        "./simulated_data/item_features.csv",
        index=False,
        columns=["item_id"] + [f"item_feature_{i+1}" for i in range(num_item_features)],
    )
    return train_df, test_df, user_feature_df, item_feature_df


train_df, test_df, user_feature_df, item_feature_df = (
    simulate_and_save_recommender_data(sparsity=0.4)
)

# %%
from jurity.recommenders import (
    BinaryRecoMetrics,
    RankingRecoMetrics,
    DiversityRecoMetrics,
)
from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.pipeline import benchmark
from mab2rec.visualization import plot_metrics_at_k
from mab2rec.visualization import plot_inter_diversity_at_k
from mab2rec.visualization import plot_recommended_counts
from mab2rec.visualization import plot_recommended_counts_by_item

metric_params = {
    "click_column": "score",
    "user_id_column": "user_id",
    "item_id_column": "item_id",
}
top_k_list = [2, 5, 10]
metrics = []
for k in top_k_list:
    metrics.append(BinaryRecoMetrics.AUC(**metric_params, k=k))
    metrics.append(BinaryRecoMetrics.CTR(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Precision(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Recall(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.NDCG(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.MAP(**metric_params, k=k))
    metrics.append(DiversityRecoMetrics.InterListDiversity(**metric_params, k=k)),
    metrics.append(
        DiversityRecoMetrics.IntraListDiversity(
            **metric_params, item_features=item_feature_df, k=k
        )
    )


recommenders = {
    "Random": BanditRecommender(learning_policy=LearningPolicy.Random()),
    "Popularity": BanditRecommender(LearningPolicy.Popularity()),
    "LinGreedy": BanditRecommender(
        learning_policy=LearningPolicy.LinGreedy(epsilon=0.1)
    ),
    "LinTS": BanditRecommender(learning_policy=LearningPolicy.LinTS()),
    "ClustersTS(K=3)": BanditRecommender(
        learning_policy=LearningPolicy.ThompsonSampling(),
        neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=3),
    ),
    "ClustersTS(K=10)": BanditRecommender(
        learning_policy=LearningPolicy.ThompsonSampling(),
        neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=10),
    ),
}
reco_to_results, reco_to_metrics = benchmark(
    recommenders,
    metrics=metrics,
    train_data="./simulated_data/train_interactions.csv",
    test_data="./simulated_data/test_interactions.csv",
    user_features="./simulated_data/user_features.csv",
    item_features="./simulated_data/item_features.csv",
    warm_start=True,
    warm_start_distance=0.75,
)

# %%
plot_metrics_at_k(reco_to_metrics, col_wrap=2)
plot_recommended_counts(
    reco_to_results, test_df, k=3, alpha=0.7, average_response=False, col_wrap=2
)
plot_recommended_counts_by_item(
    reco_to_results, k=3, top_n_items=15, normalize=True, col_wrap=2
)

# %%
# =======================================================
# Without the benchmark module train the recommenders
# =======================================================

recommenders = {
    "Random": BanditRecommender(learning_policy=LearningPolicy.Random()),
    "Popularity": BanditRecommender(LearningPolicy.Popularity()),
    "LinGreedy": BanditRecommender(
        learning_policy=LearningPolicy.LinGreedy(epsilon=0.1)
    ),
    "LinTS": BanditRecommender(learning_policy=LearningPolicy.LinTS()),
    "ClustersTS(K=3)": BanditRecommender(
        learning_policy=LearningPolicy.ThompsonSampling(),
        neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=3),
    ),
    "ClustersTS(K=10)": BanditRecommender(
        learning_policy=LearningPolicy.ThompsonSampling(),
        neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=10),
    ),
}
for rec in recommenders:
    recommenders[rec].fit(train_df)
