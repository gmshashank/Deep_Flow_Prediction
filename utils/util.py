from genericpath import exists
import math
import numpy as np
import os
import re
from PIL import Image
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
            new_img.paste(im, (s + 10) * i, s * 0)
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
