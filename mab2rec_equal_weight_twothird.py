# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
import random

# %% Create users and data
np.random.seed(123)
X, y = make_classification(
    n_samples=3000,
    n_features=20,
    n_classes=4,
    n_clusters_per_class=2,
    random_state=42,
    n_informative=10,
)
X, y = pd.DataFrame(X), pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# %%
# Initialize classifiers
conf_to_answer = np.random.rand(5)
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}
user_features = np.zeros((5, 5))
for i, (name, clf) in enumerate(classifiers.items()):
    if name in ["K-Nearest Neighbors", "Decision Tree"]:
        X_t, y_t = X_train[y_train[0] >= 2], y_train[y_train[0] >= 2]
    if name in ["Support Vector Classifier"]:
        X_t, y_t = (
            X_train[y_train[0].isin([0, 1, 3])],
            y_train[y_train[0].isin([0, 1, 3])],
        )
    else:
        X_t, y_t = X_train.copy(), y_train.copy()
    uni, count = np.unique(y_t, return_counts=True)
    for cl, cc in zip(uni, count):
        user_features[i, cl] = cc
    user_features[:, -1] = conf_to_answer
    clf.fit(X_t, y_t)

for name, clf in classifiers.items():
    score = clf.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.2f}")

# %% Build interactions

user_id, item_id, response, prediction = [], [], [], []
for idx, x in tqdm(zip(X.index, X.values), total=X.shape[0]):
    for i, (name, user) in enumerate(classifiers.items()):
        proba = user.predict_proba(x.reshape(1, -1)).max()
        ypred = user.predict(x.reshape(1, -1))
        if proba < 0.3:
            user_id.append(i)
            item_id.append(idx)
            response.append(1)
            prediction.append(ypred[0])
        else:
            user_id.append(i)
            item_id.append(idx)
            response.append(0)
            prediction.append(ypred[0])
interactions = pd.DataFrame(
    {
        "user_id": user_id,
        "item_id": item_id,
        "response": response,
        "prediction": prediction,
    }
)
interactions_train = interactions.iloc[X_train.index]
interactions_test = interactions.iloc[X_test.index]
interactions_train.to_csv("./simulated_data/train_interactions.csv", index=False)
interactions_test.to_csv("./simulated_data/test_interactions.csv", index=False)
X.to_csv("./simulated_data/item_features.csv")
user_features = pd.DataFrame(user_features)
user_features["user_id"] = user_features.index
user_features.to_csv("./simulated_data/user_features.csv", index=False)
item_eligibility = pd.DataFrame(
    {"user_id": [0, 1, 2, 3, 4], "item_id": [X_test.index.tolist() for _ in range(5)]}
)
item_eligibility.to_csv("./simulated_data/item_eligibility.csv", index=False)
# %%
from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.pipeline import train, score

rec = BanditRecommender(LearningPolicy.LinGreedy(epsilon=0.1), top_k=5)
train(
    rec,
    data="./simulated_data/train_interactions.csv",
    user_features="./simulated_data/user_features.csv",
)
df = score(
    rec,
    data="./simulated_data/test_interactions.csv",
    user_features="./simulated_data/user_features.csv",
    item_eligibility="./simulated_data/item_eligibility.csv",
)
# %%
from mab2rec.pipeline import benchmark
from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics

metric_params = {
    "click_column": "score",
    "user_id_column": "user_id",
    "item_id_column": "item_id",
}
top_k_list = [5]
metrics = []
for k in top_k_list:
    metrics.append(BinaryRecoMetrics.AUC(**metric_params, k=k))
    metrics.append(BinaryRecoMetrics.CTR(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Precision(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.Recall(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.NDCG(**metric_params, k=k))
    metrics.append(RankingRecoMetrics.MAP(**metric_params, k=k))
reco_to_results, reco_to_metrics = benchmark(
    {"Epsilon-greed": rec},
    metrics=metrics,
    train_data="./simulated_data/train_interactions.csv",
    test_data="./simulated_data/test_interactions.csv",
    user_features="./simulated_data/user_features.csv",
    item_eligibility="./simulated_data/item_eligibility.csv",
)

# %%
