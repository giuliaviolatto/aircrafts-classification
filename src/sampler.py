import numpy as np
from torch.utils.data import Sampler


class AircraftsSampler(Sampler):
    def __init__(self, df, minority_pct=0.5):
        self.df = df
        self.n = len(df)
        self.minority_pct = minority_pct

    def __iter__(self):
        minority_idxs = self.df.loc[self.df['isnato'] == 0].index.values
        majority_idxs = self.df.loc[self.df['isnato'] == 1].index.values

        minority = np.random.choice(minority_idxs, int(self.n * self.minority_pct), replace=True)
        majority = np.random.choice(majority_idxs, int(self.n * (1-self.minority_pct))+1, replace=False)

        idxs = np.hstack([minority, majority])
        np.random.shuffle(idxs)

        return iter(idxs)

    def __len__(self):
        return self.n

