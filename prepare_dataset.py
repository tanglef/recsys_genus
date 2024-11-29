# %%
import pandas as pd
import numpy as np


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


# %%
