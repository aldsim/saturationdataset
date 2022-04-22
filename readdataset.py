import numpy as np

import torch
from torch.utils.data import Dataset

import os

class ALDProfile(Dataset):

    def __init__(self, profile_file, tdose_file, tsat_file, logtime=True):

        self.profile = np.load(profile_file)
        self.tdose = np.load(tdose_file)
        self.tdose = self.tdose.reshape((len(self.tdose),1))
        self.tsat = np.load(tsat_file)

        if logtime:
            self.tdose = np.log(self.tdose)
            self.tsat = np.log(self.tsat)

        self.tdose = self.tdose.reshape(len(self.tdose),1)
        self.tsat = self.tsat.reshape((len(self.tsat),1))

    def __len__(self):
        return self.profile.shape[0]

    def __getitem__(self, idx):
        profs = torch.tensor(self.profile[idx,:],dtype=torch.float)
        tdose = torch.tensor(self.tdose[idx], dtype=torch.float)
        tsat = torch.tensor(self.tsat[idx], dtype=torch.float)
        return profs, tdose, tsat


class ALDDataset_N(ALDProfile):

    def __init__(self, N, train=True, directory="./dataset"):
        if train:
            profile_file = os.path.join(directory, "profiles_train_{}.npy".format(N))
            tdose_file = os.path.join(directory, "dose_train_{}.npy".format(N))
            tsat_file = os.path.join(directory, "sat_train_{}.npy".format(N))
        else:

            profile_file = os.path.join(directory, "profiles_test_{}.npy".format(N))
            tdose_file = os.path.join(directory, "dose_test_{}.npy".format(N))
            tsat_file = os.path.join(directory, "sat_test_{}.npy".format(N))

        super().__init__(profile_file, tdose_file, tsat_file)

class ALDDataset_4(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(4, train, directory)


class ALDDataset_5(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(5, train, directory)

class ALDDataset_8(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(8, train, directory)

class ALDDataset_10(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(10, train, directory)

class ALDDataset_16(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(16, train, directory)

class ALDDataset_20(ALDDataset_N):

    def __init__(self, train=True, directory="./dataset"):
        super().__init__(20, train, directory)


if __name__ == "__main__":

    ald_train = ALDDataset_4()
    ald_test = ALDDataset_4(train=False)
