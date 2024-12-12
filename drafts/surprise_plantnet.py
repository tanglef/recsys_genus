# %%
import pandas as pd
import numpy as np

from surprise import Dataset, NormalPredictor, Reader
from surprise.model_selection import cross_validate
from pathlib import Path

dir_ = Path(__file__).parents[0] / "data" / "recsys_interactions_minimal.parquet"


def preprocess_plantnet_data(
    logs_path,
    min_number_of_reviews=10,
    balanced_classes=False,
    use_context=True,
):
    print("preparing ratings log")
    logs = pd.read_parquet(logs_path)
    # print(pd.DataFrame(logs["obs_id"].value_counts())["obs_id"])
    obs_counts = logs["obs_id"].value_counts()
    obs_to_keep = (
        pd.DataFrame(obs_counts)
        .loc[pd.DataFrame(obs_counts)["count"] >= min_number_of_reviews]
        .index
    )
    logs = logs.loc[logs["obs_id"].isin(obs_to_keep)]
    logs["rating"] = logs["interaction_type"].map(
        {"determination": 5, "quality": 1, "noplant": 1, "organ": 2, "malformed": 1}
    )
    logs = (
        logs[["user_id", "obs_id", "rating"]]
        .groupby(["user_id", "obs_id"])
        .sum()
        .reset_index()
    )
    logs.loc[logs["rating"] > 10, "rating"] = 10

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
    return logs, obs_info, user_info


from surprise import accuracy, Dataset, SVD
from surprise.model_selection import train_test_split

logs, *_ = preprocess_plantnet_data(dir_, min_number_of_reviews=2, use_context=False)
logs.drop("t", inplace=True, axis=1)
logs.columns = ["userID", "itemID", "rating"]
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(logs[["userID", "itemID", "rating"]], reader)
algo = SVD()
trainset, testset = train_test_split(data, test_size=0.25)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
# %%
test_subject_iid = trainset.to_inner_uid("101028285")
test_subject_ratings = trainset.ur[test_subject_iid]

# %%
from collections import defaultdict


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=8)
# Precision and recall can then be averaged over all users
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))
# %%
from surprise.prediction_algorithms.matrix_factorization import NMF

algo = NMF(lr_bu=0.5, lr_bi=0.5)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
# Print the recommended items for each user
# for uid, user_ratings in top_n.items():
#     print(uid, [iid for (iid, _) in user_ratings])
precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=8)
# Precision and recall can then be averaged over all users
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))
# %%
