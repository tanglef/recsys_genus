# %%
import pandas as pd
import numpy as np
from mab2rec.utils import print_interaction_stats
import seaborn as sns
import matplotlib.pyplot as plt
from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics
from mab2rec.pipeline import benchmark

sns.set_style("whitegrid")
import warnings
from mab2rec.visualization import plot_metrics_at_k
from mab2rec.visualization import plot_intra_diversity_at_k
from crowdkit.aggregation import Wawa
from tqdm.auto import tqdm


def get_num_people_by_age_category(df):
    df["age_group"] = pd.cut(
        x=df["age"], bins=np.arange(90, step=10), labels=np.arange(90, step=10)[:-1]
    )
    return df


full_data = pd.read_csv(
    "./data/ml-100k/u.data",
    names=["user_id", "item_id", "rating"],
    usecols=(0, 1, 2),
    sep="\t",
)
user_features = pd.read_csv(
    "./data/ml-100k/u.user",
    names=["user_id", "age", "genre", "occupation", "other"],
    sep="|",
)
user_features.drop(["genre", "other"], axis=1, inplace=True)
user_features = pd.get_dummies(user_features, columns=["occupation"], drop_first=True)
user_features = get_num_people_by_age_category(user_features)
user_features.drop(["age"], axis=1, inplace=True)
user_features = pd.get_dummies(user_features, columns=["age_group"], drop_first=True)
names = ["item_id", "name", "date", "other", "url"] + [f"genre_{i}" for i in range(19)]
item_features = pd.read_csv(
    "./data/ml-100k/u.item", names=names, sep="|", encoding="latin-1"
)
item_features.drop(["other", "name", "date", "url"], axis=1, inplace=True)
item_features.to_csv("./data/ml-100k/item_features.csv", index=False)
user_features.to_csv("./data/ml-100k/user_features.csv", index=False)
test_data = pd.read_csv(
    "./data/ml-100k/ua.test",
    delimiter="\t",
    header=None,
    names=["user_id", "item_id", "rating"],
    usecols=(0, 1, 2),
)
movies_to_keep = pd.DataFrame(full_data["item_id"].value_counts().reset_index()).loc[
    pd.DataFrame(full_data["item_id"].value_counts()).reset_index()["count"] >= 20
]["item_id"]
full_data = full_data.loc[full_data["item_id"].isin(movies_to_keep)]
test_data = test_data.loc[test_data["item_id"].isin(movies_to_keep)]
train_data = (
    pd.merge(
        full_data,
        test_data,
        on=["user_id", "item_id", "rating"],
        how="outer",
        indicator=True,
    )
    .query("_merge != 'both'")
    .drop("_merge", axis=1)
    .reset_index(drop=True)
)
train_data["rating"] = train_data["rating"] > 4
test_data["rating"] = test_data["rating"] > 4
train_data.columns = ["user_id", "item_id", "response"]
test_data.columns = ["user_id", "item_id", "response"]
test_data.to_csv("./data/ml-100k/test_data.csv", index=False)
train_data.to_csv("./data/ml-100k/train_data.csv", index=False)
dir_data = "./data/ml-100k/"

train_data = dir_data + "train_data.csv"
test_data = dir_data + "test_data.csv"
user_data = "./../mab2rec/data/" + "features_user.csv"
user_df = pd.read_csv(user_data)
item_data = dir_data + "item_features.csv"
item_df = pd.read_csv(dir_data + "item_features.csv")
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(dir_data + "test_data.csv")
user_df = pd.read_csv(dir_data + "user_features.csv")


def get_train_recommenders(train_data, test_data, user_data):
    # Set of recommenders
    recommenders = {
        "Random": BanditRecommender(learning_policy=LearningPolicy.Random()),
        # "LinGreedy": BanditRecommender(
        #     learning_policy=LearningPolicy.LinGreedy(epsilon=0.1)
        # ),
        # "LinUCB": BanditRecommender(
        #     learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=1)
        # ),
        # "TS": BanditRecommender(learning_policy=LearningPolicy.ThompsonSampling()),
        # "LinTS": BanditRecommender(learning_policy=LearningPolicy.LinTS()),
        # "ClustersTS(L=5)": BanditRecommender(
        #     learning_policy=LearningPolicy.ThompsonSampling(),
        #     neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=5),
        # ),
        # "ClustersTS(L=10)": BanditRecommender(
        #     learning_policy=LearningPolicy.ThompsonSampling(),
        #     neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=10),
        # ),
        # "ClustersTS(L=20)": BanditRecommender(
        #     learning_policy=LearningPolicy.ThompsonSampling(),
        #     neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=20),
        # ),
        # "ClustersTS(L=30)": BanditRecommender(
        #     learning_policy=LearningPolicy.ThompsonSampling(),
        #     neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=30),
        # ),
        # "LSHNearest": BanditRecommender(
        #     learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.1),
        #     neighborhood_policy=NeighborhoodPolicy.LSHNearest(5, 3),
        # ),
    }
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
    reco_to_results, reco_to_metrics = benchmark(
        recommenders,
        metrics=metrics,
        train_data=train_data,
        test_data=test_data,
        user_features=user_data,
    )
    return recommenders, metrics, reco_to_results, reco_to_metrics


user_df = pd.read_csv(user_data)
item_df = pd.read_csv(item_data)
user_df = user_df.drop("user_id", axis=1)
by_genre_df_train = {"user_id": [], "item_id": [], "reward": []}
item_df.index = item_df["item_id"]
for i, row in train_df.iterrows():
    item_genre = np.nonzero(item_df.loc[row[1]])[0][1:] - 1
    for genre in item_genre:
        by_genre_df_train["user_id"].append(row[0])
        by_genre_df_train["item_id"].append(genre)
        by_genre_df_train["reward"].append(row[2])
by_genre_df_train = pd.DataFrame(by_genre_df_train)
by_genre_df_train = (
    by_genre_df_train.groupby(["user_id", "item_id"]).count().reset_index()
)
by_genre_df_train["response"] = by_genre_df_train["reward"] > 5
by_genre_df_test = {"user_id": [], "item_id": [], "reward": []}
item_df.index = item_df["item_id"]
for i, row in test_df.iterrows():
    item_genre = np.nonzero(item_df.loc[row[1]])[0][1:] - 1
    for genre in item_genre:
        by_genre_df_test["user_id"].append(row[0])
        by_genre_df_test["item_id"].append(genre)
        by_genre_df_test["reward"].append(row[2])
by_genre_df_test = pd.DataFrame(by_genre_df_test)
by_genre_df_test = (
    by_genre_df_test.groupby(["user_id", "item_id"]).count().reset_index()
)
by_genre_df_test["response"] = by_genre_df_test["reward"] > 5
by_genre_df_train.to_csv(dir_data + "train_by_genre.csv", index=False)
by_genre_df_test.to_csv(dir_data + "test_by_genre.csv", index=False)

recommenders, metrics, reco_to_results, reco_to_metrics = get_train_recommenders(
    dir_data + "train_by_genre.csv", dir_data + "test_by_genre.csv", user_data
)
# %%
from crowdrec import CrowdRecOffline

warnings.filterwarnings("ignore")
np.random.seed(123)
budgets = np.arange(
    100,
    2001,
    step=200,
)
pools = {key: [] for key in by_genre_df_test.item_id.unique()}
for key in pools.keys():
    pools[key].append(
        item_df["item_id"][np.nonzero(item_df[f"genre_{key}"])[0] + 1].values
    )
    pools[key] = pools[key][0]

for seed in tqdm(range(20), leave=True):
    for name, rec in recommenders.items():
        path_save = f"./outputs/results_{name}_{seed}.csv"
        CREC = CrowdRecOffline(
            recommender=rec,
            name=name,
            test_data=test_df,
            user_features=user_df,
            budgets=budgets,
            seed=seed,
            n_classes=2,
            output_file=path_save,
            pools=pools,
            genre_df_test=by_genre_df_test,
        )
        CREC.run()

# %%
