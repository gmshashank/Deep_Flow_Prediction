import numpy as np
import os
import random
import torch

fixedAirfoilNormalization = True  # use fixed max values fro dim-less airfoil data
makeDimLess = True  # make data dimensionless
removePOffset = True  # remove constant offsets from pressure channels


def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


def LoaderNormalizer(data, isTest=False, shuffle=0, dataProp=None):

    mode = "TRAIN"

    if dataProp is None:
        # load single directory
        files = os.listdir(data.dataDir)
        files.sort()
        for i in range(shuffle):
            random.shuffle(files)
        if isTest:
            print("Reducing data to load for tests")
            files = files[0 : min(10, len(files))]
        data.totalLength = len(files)
        data.inputs = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))

        for i, file in enumerate(files):
            npfile = np.load(data.dataDir + file)
            d = npfile["a"]
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]
        print(f"Number of data loaded: {len(data.inputs)}")
    else:
        # load from folders reg,sup and shear
        data.totalLength = int(dataProp[0])
        data.inputs = np.empty((data.totalLength, 3, 128, 128))
        data.targets = np.empty((data.totalLength, 3, 128, 128))

        files_reg = os.listdir(data.dataDir + "reg/")
        files_reg.sort()
        files_sup = os.listdir(data.dataDir + "sup/")
        files_sup.sort()
        files_shear = os.listdir(data.dataDir + "shear/")
        files_shear.sort()

        for i in range(shuffle):
            random.shuffle(files_reg)
            random.shuffle(files_sup)
            random.shuffle(files_shear)

        temp_1, temp_2 = 0, 0
        for i in range(data.totalLength):
            if i >= (1 - dataProp[3]) * dataProp[0]:
                npfile = np.load(data.dataDir + "shear/" + files_shear[i - temp_2])
                d = npfile["a"]
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
            elif i >= (dataProp[1]) * dataProp[0]:
                npfile = np.load(data.dataDir + "sup/" + files_sup[i - temp_1])
                d = npfile["a"]
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
                temp_2 = i + 1
            else:
                npfile = np.load(data.dataDir + "reg/" + files_reg[i])
                d = npfile["a"]
                data.inputs[i] = d[0:3]
                data.targets = d[3:6]
                temp_1 = i + 1
                temp_2 = i + 1
        print(f"Number od data loaded (reg={temp_1}, sup={temp_2}, shear={i+1-temp_2}")

    # Normalization of Training Data

    if removePOffset:
        for i in range(data.totalLength):
            data.targets[i, 0, :, :] -= np.mean(data.targets[i, 0, :, :])  # remove offset
            data.targets[i, 0, :, :] -= data.targets[i, 0, :, :] * data.inputs[i, 2, :, :]  # pressure * mask

    # make dimensionless based on current dataset
    if makeDimLess:
        for i in range(data.totalLength):
            v_norm = (
                np.max(np.abs(data.inputs[i, 0, :, :])) ** 2 + np.max(np.abs(data.inputs[i, 1, :, :])) ** 2
            ) ** 0.5
            data.targets[i, 0, :, :] /= v_norm ** 2
            data.targets[i, 1, :, :] /= v_norm
            data.targets[i, 2, :, :] /= v_norm

    if fixedAirfoilNormalization:
        # hard coded maxima , inputs dont change
        data.max_inputs_0 = 100.0
        data.max_inputs_1 = 38.12
        data.max_inputs_2 = 1.0

        # targets depend on Normalization
        if makeDimLess:
            data.max_targets_0 = 4.65
            data.max_targets_1 = 2.04
            data.max_targets_2 = 2.37
            print(f"Using fixed maxima [{data.max_targets_0}, {data.max_targets_1}, {data.max_targets_2}]")
        else:  # full range
            data.max_targets_0 = 40000.0
            data.max_targets_1 = 200.0
            data.max_targets_2 = 216.0
            print(f"Using fixed maxima [{data.max_targets_0}, {data.max_targets_1}, {data.max_targets_2}]")

    else:  # Use current max values from loaded data
        data.max_inputs_0 = find_absmax(data, 0, 0)
        data.max_inputs_1 = find_absmax(data, 0, 1)
        data.max_inputs_2 = find_absmax(data, 0, 2)
        print(f"Maxima inputs [{data.max_inputs_0}, {data.max_inputs_1}, {data.max_inputs_2}]")

        data.max_targets_0 = find_absmax(data, 1, 0)
        data.max_targets_1 = find_absmax(data, 1, 1)
        data.max_targets_2 = find_absmax(data, 1, 2)
        print(f"Maxima targets [{data.max_targets_0}, {data.max_targets_1}, {data.max_targets_2}]")

    data.inputs[:, 0, :, :] *= 1.0 / data.max_inputs_0
    data.inputs[:, 1, :, :] *= 1.0 / data.max_inputs_1

    data.targets[:, 0, :, :] *= 1.0 / data.max_targets_0
    data.targets[:, 1, :, :] *= 1.0 / data.max_targets_1
    data.targets[:, 2, :, :] *= 1.0 / data.max_targets_2

    # Normalization of Test Data
    if isTest:
        mode = "TEST"
        files = os.listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))

        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile["a"]
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]

        if removePOffset:
            for i in range(data.totalLength):
                data.targets[i, 0, :, :] -= np.mean(data.targets[i, 0, :, :])  # remove Offset
                data.targets[i, 0, :, :] -= data.targets[i, 0, :, :] * data.inputs[i, 2, :, :]  # pressure * mask

        if makeDimLess:
            for i in range(len(files)):
                v_norm = (
                    np.max(np.abs(data.inputs[i, 0, :, :])) ** 2 + np.max(np.abs(data.inputs[i, 1, :, :])) ** 2
                ) ** 0.5
                data.targets[i, 0, :, :] /= v_norm ** 2
                data.targets[i, 1, :, :] /= v_norm
                data.targets[i, 2, :, :] /= v_norm

        data.inputs[:, 0, :, :] *= 1.0 / data.max_inputs_0
        data.inputs[:, 1, :, :] *= 1.0 / data.max_inputs_1

        data.targets[:, 0, :, :] *= 1.0 / data.max_targets_0
        data.targets[:, 1, :, :] *= 1.0 / data.max_targets_1
        data.targets[:, 2, :, :] *= 1.0 / data.max_targets_2

    print(f"Data Stats: {mode}")
    print(
        f"Input \t mean {np.mean(np.abs(data.inputs),keepdims=False)} \t max {np.max(np.abs(data.inputs),keepdims=False)}"
    )
    print(
        f"targets \t mean {np.mean(np.abs(data.targets),keepdims=False)} \t max {np.max(np.abs(data.targets),keepdims=False)}"
    )

    return data


class TurbDataset(torch.utils.data.Dataset):
    TRAIN = 0
    TEST = 2

    def __init__(
        self, dataProp=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0
    ):
        global makeDimLess, removePOffset

        if not (mode == self.TRAIN or mode == self.TEST):
            print(f"Error - TurbDataset invalid mode = {mode}")
            exit(1)

        if normMode == 1:
            print("POffset off")
            removePOffset = False
        elif normMode == 2:
            print("POffset off and DimLess off")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest
        self = LoaderNormalizer(self, isTest=(mode == self.TEST), dataProp=dataProp, shuffle=shuffle)

        if not self.mode == self.TEST:
            targetLength = self.totalLength - min(int(self.totalLength * 0.2), 400)
            self.validInputs = self.inputs[targetLength:]
            self.validTargets = self.targets[targetLength:]
            self.validLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0, :, :] /= 1.0 / self.max_targets_0
        a[1, :, :] /= 1.0 / self.max_targets_1
        a[2, :, :] /= 1.0 / self.max_targets_2

        if makeDimLess:
            a[0, :, :] *= v_norm ** 2
            a[1, :, :] *= v_norm
            a[2, :, :] *= v_norm

        return a


class ValidDataset(TurbDataset):
    def __init__(self, dataset):
        self.inputs = dataset.validInputs
        self.targets = dataset.validTargets
        self.totalLength = dataset.validLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
