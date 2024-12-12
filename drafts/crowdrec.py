from abc import ABC, abstractclassmethod
import numpy as np
import pandas as pd


class CrowdRec(ABC):

    @abstractclassmethod
    def check_quality(self):
        pass

    @abstractclassmethod
    def aggregate(self):
        pass

    @abstractclassmethod
    def recommend(self):
        pass

    @abstractclassmethod
    def update_results(self):
        pass

    @abstractclassmethod
    def save_results(self):
        pass

    @abstractclassmethod
    def run(self):
        pass


class CrowdRecOffline(CrowdRec):

    def __init__(
        self,
        recommender,
        name,
        test_data,
        user_features,
        budgets,
        genre_df_test=None,
        n_classes=2,
        pools=None,
        output_file="./outputs/results.csv",
        seed=123,
    ):
        np.random.seed(seed)
        self.results = {
            "seed": [],
            "annotated": [],
            "name": [],
            "skipped": [],
            "budget": [],
            "recommender": [],
        }
        self.output_file = output_file
        self.n_classes = n_classes
        self.name = name
        self.seed = seed
        self.test_data = test_data
        self.user_features = user_features
        self.genre_df_test = genre_df_test
        self.n_item_test = test_data["item_id"].max() + 1
        if type(budgets) is not list and type(budgets) is not np.ndarray:
            self.budget = [budgets]
            self.budgets = self.budget
        else:
            self.budgets = budgets
            self.budget = budgets[-1]
        self.quality_threshold = np.zeros(self.n_item_test)
        if self.genre_df_test is not None:
            self.users = self.genre_df_test["user_id"].unique()
            self.test_genras = self.genre_df_test.item_id.unique()
            self.pools = pools
        else:
            self.users = self.test_data["user_id"].unique()
            self.test_genras = None
        self.recommender = recommender
        self.current_budget = 0

    def check_quality(self, votes, rule="TwoThird"):
        if rule == "TwoThird":
            sum_ = votes.sum(axis=1, keepdims=True)
            mask1 = np.max(votes / sum_, axis=1) >= 2 / 3
            mask2 = (sum_ > 2).flatten()
            mask = np.logical_and(mask1, mask2)
            self.quality_threshold[mask] = 1
            self.quality_threshold[~mask] = 0
        else:
            raise NotImplementedError

    def aggregate(self, votes, user, item, label, rule="TwoThird"):
        if rule == "TwoThird":
            votes[item, label.iloc[0]] += 1
            return votes
        else:
            raise NotImplementedError

    def recommend(self, user):
        if self.recommender.mab.is_contextual:
            genus_or_item_recs = self.recommender.predict_expectations(
                self.user_features.iloc[user - 1]
            )
        else:
            genus_or_item_recs = self.recommender.predict_expectations()
        return dict(sorted(genus_or_item_recs.items(), key=lambda item: item[1]))

    def update_results(self):
        self.results["seed"].append(self.seed)
        self.results["name"].append(self.name)
        self.results["annotated"].append(self.quality_threshold.sum())
        self.results["skipped"].append(self.skipped)
        self.results["budget"].append(self.current_budget)

    def save_results(self, output_file="./outputs/results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)

    def run_mortal_rec(self, item_recs, user_interactions, user):
        user_items = user_interactions["item_id"].unique()
        if self.quality_threshold[user_items].sum() == len(user_items):
            return "No arm is available (all valid)"
        for item, _ in item_recs.items():
            if self.quality_threshold[item] == 0 and item not in self.seen[user]:
                # arm is available
                if item in user_items:
                    # arm is annotated
                    return item
                else:
                    # arm is not annotated
                    return None
            else:
                # arm is dead
                continue

    def run_crowd_rec(self, genus_recs, user_interactions, user):
        user_items = user_interactions["item_id"].unique()
        if self.quality_threshold[user_items].sum() == len(user_items):
            return "No item is available (all valid)"
        for genus, _ in genus_recs.items():
            # we only look at the first genus
            possible_items = self.pools[genus]
            temp_ = user_interactions.loc[
                user_interactions["item_id"].isin(possible_items)
            ]
            if len(temp_.index) == 0:
                # user never interacted with this genus
                return None
            else:
                temp_ = temp_.loc[~(temp_["item_id"].isin(self.seen[user]))]
                if len(temp_.index) > 0:
                    keep_looking = True
                    while not keep_looking:
                        item = np.random.choice(temp_["item_id"])
                        if self.quality_threshold[item] == 0:
                            keep_looking = False
                    return item
                else:
                    # user already saw every item
                    return "No item is available (already seen)"

    def run(self):
        keep_recommending = True
        self.skipped = 0
        votes = np.zeros((self.n_item_test, self.n_classes))
        self.seen = {u: [] for u in self.users}
        while keep_recommending:
            user = np.random.choice(self.users)
            user_interactions = self.test_data[self.test_data["user_id"] == user]
            genus_or_item_recs = self.recommend(user)
            if self.test_genras is None:
                # in the mortal arm setting
                item = self.run_mortal_rec(genus_or_item_recs, user_interactions, user)
            else:
                # in the genre-based setting
                item = self.run_crowd_rec(genus_or_item_recs, user_interactions, user)
            if item is None:
                # Unaligned recommendation
                self.skipped += 1
                self.current_budget += 1
            elif type(item) is str:
                # No arm is available or everything already seen
                continue
            else:
                # aligned recommendation
                self.current_budget += 1
                vote = user_interactions[user_interactions["item_id"] == item][
                    "response"
                ]
                votes = self.aggregate(votes, user, item, vote, rule="TwoThird")
                self.check_quality(votes, rule="TwoThird")
            if self.current_budget in self.budgets:
                self.update_results()

            keep_recommending = self.current_budget < self.budget
        self.save_results(output_file=self.output_file)
