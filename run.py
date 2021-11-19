import argparse
from typing import Generator
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn

from datasets.dataset import TurbDataset, ValidDataset
from models import Generator, weights_init
from utils import computeLR, makeDirs, imageOut, plot_loss


def main(args=None):

    seed = args.manual_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
        torch.cuda.manual_seed_all(seed)

    batch_size = args.batch_size
    lrG = args.lr
    channelExponent = args.channel_exponent
    epochs = args.num_epochs
    dataDir = args.data_dir + "train/"
    dataDirTest = args.data_dir + "test/"

    image_height = 128
    image_width = 128
    image_channels = 3
    beta1 = 0.5
    beta2 = 0.999
    weight_decay = 0.0
    decayLR = True
    prop = None
    saveL1 = False
    dropout = 0.0

    # Data setup

    # print(len(os.listdir(dataDir)))
    # print(len(os.listdir(dataDirTest)))
    training_data = TurbDataset(prop, shuffle=1, mode=0, dataDir=dataDir, dataDirTest=dataDirTest, normMode=1)
    # training_data.to(device=device,dtype=torch.float)
    # test_data = TurbDataset(prop, shuffle=1, mode=2, dataDir=dataDir, dataDirTest=dataDirTest, normMode=1)

    trainLoader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Training batches: {len(trainLoader)}")

    dataValidation = ValidDataset(training_data)
    validLoader = torch.utils.data.DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"Validation batches: {len(validLoader)}")

    # Training Setup

    netG = Generator(channelExponent=channelExponent, dropout=dropout)
    nn_parameters = filter(lambda p: p.requires_grad, netG.parameters())
    params = sum([np.prod(p.size()) for p in nn_parameters])

    print(f"Trainable params: {params}.")

    # netG.double()       #initialize a model’s weights with DoubleTensors
    netG.apply(weights_init)
    netG.to(device)

    criterionL1 = nn.L1Loss()
    criterionL1.to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, beta2), weight_decay=weight_decay)

    inputs = torch.FloatTensor(batch_size, image_channels, image_height, image_width)
    inputs = inputs.to(device)

    targets = torch.FloatTensor(batch_size, image_channels, image_height, image_width)
    targets = targets.to(device)

    # print(device)
    # print(inputs.device)
    # print(targets.device)

    history_L1 = []
    history_L1val = []

    print("Training the network on {device}")

    for epoch in range(epochs):
        netG.train()
        L1_accum = 0.0
        for i, traindata in enumerate(trainLoader, 0):
            inputs_cpu, targets_cpu = traindata
            inputs_cpu, targets_cpu = inputs_cpu.float().to(device), targets_cpu.float().to(device)
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

            # compute LR decay
            if decayLR:
                currLR = computeLR(epoch, epochs, lrG * 0.1, lrG)
                if currLR < lrG:
                    for g in optimizerG.param_groups:
                        g["lr"] = currLR

            netG.zero_grad()
            gen_out = netG(inputs)

            lossL1 = criterionL1(gen_out, targets)
            lossL1.backward()
            optimizerG.step()
            L1_accum += lossL1.item()

        # validation
        netG.eval()
        with torch.no_grad():
            L1val_accum = 0.0
            for i, validdata in enumerate(validLoader, 0):
                inputs_cpu, targets_cpu = validdata
                inputs_cpu, targets_cpu = inputs_cpu.float().to(device), targets_cpu.float().to(device)
                inputs.resize_as_(inputs_cpu).copy_(inputs_cpu)
                targets.resize_as_(targets_cpu).copy_(targets_cpu)

                outputs = netG(inputs)
                outputs_cpu = outputs.data.cpu().numpy()

                lossL1val = criterionL1(outputs, targets)
                L1val_accum += lossL1val.item()

                if i == 0:
                    input_ndarray = inputs_cpu.cpu().numpy()[0]
                    v_norm = (
                        np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :])) ** 2
                    ) ** 0.5
                    outputs_denormalized = training_data.denormalize(outputs_cpu[0], v_norm)
                    targets_denormalized = training_data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
                    makeDirs(["results_train"])
                    imageOut(
                        f"results_train/epoch{epoch}_{i}", outputs_denormalized, targets_denormalized, saveTargets=True
                    )

        history_L1.append(L1_accum / len(trainLoader))
        history_L1val.append(L1val_accum / len(validLoader))

        print(f"Epoch: {epoch}, L1 train: {history_L1[-1]}, L1 validation: {history_L1val[-1]}")

        if (epoch + 1) % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    ",odel_state_dict": netG.state_dict(),
                    "optimizer_state_dict": optimizerG.state_dict(),
                    "loss": lossL1,
                },
                f"newtork_expo_{channelExponent}_{epoch}.pth",
            )

    print("Saving the trained Model")
    torch.save(netG.state_dict(), f"modelG_{channelExponent}.pth")
    traced_model = torch.jit.trace(netG.to("cpu"), torch.randn(1, 3, 128, 128))
    traced_model.save("modelG_traced.pth")
    plot_loss(history_L1, history_L1val)
    print("Completed Training and saved trained model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--channel_exponent", type=int, default=5, help="Channel Exponent")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to data directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.0006, help="Learning Rate")
    parser.add_argument("--manual_seed", type=int, default=42, help="Initial Seed")
    args = parser.parse_args()
    main(args)
