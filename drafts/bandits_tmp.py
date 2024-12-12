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
    dir_, min_number_of_reviews=10, min_number_of_participations=100, use_context=False
)
logs.drop("t", inplace=True, axis=1)
logs.columns = ["user_id", "item_id", "rating"]


# train_data, test_data = stratified_split(logs)
# print("Train Data:\n", train_data)
# print("Test Data:\n", test_data)


class LinUCB:
    def __init__(self, n_users, n_items, n_features, alpha=0.1):
        self.alpha = alpha
        self.n_features = n_features

        # Initialize user and item matrices
        self.user_features = np.random.normal(
            0, 1, (n_users, n_features)
        )  # User feature vectors
        self.item_features = np.random.normal(
            0, 1, (n_items, n_features)
        )  # Item feature vectors

        # Initialize A (identity matrix for each user) and b (zero vector for each user)
        self.A = {user_id: np.identity(n_features) for user_id in range(n_users)}
        self.b = {user_id: np.zeros(n_features) for user_id in range(n_users)}

    def predict(self, user_id, item_id):
        x = self.user_features[user_id] * self.item_features[item_id]
        A_inv = np.linalg.inv(self.A[user_id])
        theta = A_inv.dot(self.b[user_id])
        expected_reward = theta.dot(x)
        ucb = self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))

        return expected_reward + ucb

    def update(self, user_id, item_id, rating):
        x = self.user_features[user_id] * self.item_features[item_id]
        self.A[user_id] += np.outer(x, x)
        self.b[user_id] += rating * x

    def recommend(self, user_id, item_ids):
        scores = [self.predict(user_id, item_id) for item_id in item_ids]
        return item_ids[np.argmax(scores)]


class EpsilonGreedyBandit:
    def __init__(self, n_users, n_items, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = np.zeros((n_users, n_items))
        self.counts = np.zeros((n_users, n_items))

    def recommend(self, user_id, item_ids):
        if random.random() < self.epsilon:
            return random.choice(item_ids)
        return item_ids[np.argmax(self.q_values[user_id, item_ids])]

    def update(self, user_id, item_id, reward):
        self.counts[user_id, item_id] += 1
        alpha = 1 / self.counts[user_id, item_id]
        self.q_values[user_id, item_id] += alpha * (
            reward - self.q_values[user_id, item_id]
        )

    def predict(self, user_id, item_id=None, all_items=False):
        # Predict the expected reward (average observed rating) for the given user-item pair
        if all_items:
            return self.q_values[user_id, :]
        return self.q_values[user_id, item_id]


def evaluate_bandit(bandit, data, k=5):
    precision_at_k = 0
    recall_at_k = 0
    cumulative_reward = [0]
    n_interactions = len(data)

    for _, row in tqdm(data.iterrows(), total=len(data)):
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])
        actual_rating = row["rating"]

        # Get top-k recommendations
        item_ids = data["item_id"].unique()
        recommended_item = bandit.recommend(user_id, item_ids)

        # Update bandit with observed reward
        bandit.update(user_id, recommended_item, actual_rating)

        # # Evaluate top-k metrics (simple version)
        # top_k_items = np.argsort(bandit.predict(user_id, all_items=True))[-k:]

        # if item_id in top_k_items:
        #     top_k_precision += 1
        #     top_k_recall += 1

        # Update cumulative reward
        cumulative_reward.append(actual_rating)

    # precision_at_k = top_k_precision / n_interactions
    # recall_at_k = top_k_recall / n_interactions
    return precision_at_k, recall_at_k, cumulative_reward


# Initialize models
n_users = logs["user_id"].nunique()
n_items = logs["item_id"].nunique()
n_features = 5  # Example number of features
alpha = 0.1
epsilon = 0.1

linucb_model = LinUCB(n_users, n_items, n_features, alpha)
epsilon_greedy_model = EpsilonGreedyBandit(n_users, n_items, epsilon)

# Track cumulative rewards
linucb_rewards = []
epsilon_greedy_rewards = []

# Evaluate models on test data
for bandit, rewards in zip(
    [epsilon_greedy_model, linucb_model],
    [epsilon_greedy_rewards, linucb_rewards],
):
    print("Running:", bandit.__class__.__name__)
    precision, recall, cumulative_reward = evaluate_bandit(bandit, logs)
    print(
        f"Model {bandit.__class__.__name__} - Precision@K: {precision}, Recall@K: {recall}"
    )
    break

# Plotting cumulative reward comparison
# plt.plot(linucb_rewards, label="LinUCB")
plt.plot(
    range(len(cumulative_reward)), np.cumsum(cumulative_reward), label="Epsilon-Greedy"
)
plt.yscale("log")
plt.xlabel("Interaction")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")
plt.legend()
plt.show()
