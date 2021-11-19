from genericpath import exists
import math
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

# append line to log file
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.wrtite(line + "\n")
    f.close()
    if doPrint:
        print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()


def plot_loss(history_L1, history_L1val):
    l1train = np.asarray(history_L1)
    l1vali = np.asarray(history_L1val)

    plt.figure()
    plt.plot(np.arange(l1train.shape[0]), l1train, "b", label="Training loss")
    plt.plot(np.arange(l1vali.shape[0]), l1vali, "g", label="Validation loss")
    plt.legend()
    plt.show()


def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.0

    fmin = 0.0
    fmax = 6.0
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)

    s = outputs.shape[1]
    if saveMontage:
        new_img = Image.new("RGB", ((s + 10) * 3, s * 2), color=(255, 255, 255))
        BW_img = Image.new("RGB", ((s + 10) * 3, s * 3), color=(255, 255, 255))

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else:
            outputs[i] -= -1.0
            targets[i] -= -1.0
            outputs[i] /= 2.0
            targets[i] /= 2.0

        if not saveMontage:
            suffix = ""
            if i == 0:
                suffix = "_pressure"
            elif i == 1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512, 512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512, 512))
                im.save(filename + suffix + "_target.png")

        else:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_img.paste(im, ((s + 10) * i, s * 0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_img.paste(im, ((s + 10) * i, s * 1))

            im = Image.fromarray(targets[i] * 256.0)
            BW_img.paste(im, ((s + 10) * i, s * 0))
            im = Image.fromarray(outputs[i] * 256.0)
            BW_img.paste(im, ((s + 10) * i, s * 1))
            im = Image.fromarray(np.abs(targets[i] - outputs[i]) * 10.0 * 256.0)
            BW_img.paste(im, ((s + 10) * i, s * 2))

    if saveMontage:
        new_img.save(filename + ".png")
        BW_img.save(filename + "_bw.png")


def imageOut(filename, _outputs, saveTargets=True, normalize=False):
    outputs = np.copy(_outputs)
    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        min_value = np.min(outputs[i])
        max_value = np.max(outputs[i])
        if normalize:
            outputs[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
        else:  # from -1,1 to 0,1
            outputs[i] -= -1.0
            outputs[i] /= 2.0

        suffix = ""
        if i == 0:
            suffix = "_pressure"
        elif i == 1:
            suffix = "_velX"
        else:
            suffix = "_velY"

        im = Image.fromarray(cm.magma(outputs[i], bytes=True))
        im = im.resize((128, 128))
        im.save(filename + suffix + "_pred.png")


def saveOutput(output_arr, target_arr):
    if target_arr is None:
        imageOut("./results/result", output_arr)
    else:
        imageOut(
            "./results/result", output_arr, target_arr, normalize=False, saveMontage=True
        )  # write normalized with error


class InputData:
    def __init__(self, npz_arr, removePOffset=True, makeDimLess=True):
        self.input = None
        self.target = None

        self.max_inputs_0 = 100.0
        self.max_inputs_1 = 38.12
        self.max_inputs_2 = 1.0

        self.max_targets_0 = 4.65
        self.max_targets_1 = 2.04
        self.max_targets_2 = 2.37

        if npz_arr.shape[0] >= 3:
            self.input = npz_arr[0:3]
        if npz_arr.shape[0] == 6:
            self.target = npz_arr[3:6]

        self.removePOffset = removePOffset
        self.makeDimLess = makeDimLess

        self.normalize()

    def normalize(self):
        if self.target is not None:
            if self.removePOffset:
                self.target[0, :, :] -= np.mean(self.target[0, :, :])  # remove offset
                self.target[0, :, :] -= self.target[0, :, :] * self.input[2, :, :]  # pressure * mask

            if self.makeDimLess:
                v_norm = (np.max(np.abs(self.input[0, :, :])) ** 2 + np.max(np.abs(self.input[1, :, :])) ** 2) ** 0.5
                self.target[0, :, :] /= v_norm ** 2
                self.target[1, :, :] /= v_norm
                self.target[2, :, :] /= v_norm

            self.target[0, :, :] *= 1.0 / self.max_targets_0
            self.target[1, :, :] *= 1.0 / self.max_targets_1
            self.target[2, :, :] *= 1.0 / self.max_targets_2

        if self.input is not None:
            self.input[0, :, :] *= 1 / self.max_inputs_0
            self.input[1, :, :] *= 1 / self.max_inputs_1

    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0, :, :] /= 1.0 / self.max_targets_0
        a[1, :, :] /= 1.0 / self.max_targets_1
        a[2, :, :] /= 1.0 / self.max_targets_2

        if self.makeDimLess:
            a[0, :, :] *= v_norm ** 2
            a[1, :, :] *= v_norm
            a[2, :, :] *= v_norm
        return a
