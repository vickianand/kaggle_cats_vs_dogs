import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision import transforms

from model.models import TwoConvOnePool, VggTypeNet

import warnings

# warnings.filterwarnings("ignore")


def train(
    train_folder,
    device,
    model_path="data/model/",
    batch_norm=False,
    decreasing_lr=False,
):

    os.makedirs(model_path, exist_ok=True)

    # Augmentation and Normalization

    transform_set = {
        "dataset_train": transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     [0.4899, 0.4599, 0.4163], [0.2521, 0.2451, 0.2473]
                # ),
            ]
        ),
        "dataset_validn": transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     [0.4899, 0.4599, 0.4163], [0.2521, 0.2451, 0.2473]
                # ),
            ]
        ),
    }

    dataset = ImageFolder(root=train_folder, transform=transform_set["dataset_train"])
    # dataset = torch.utils.data.Subset(dataset, indices=range(200))

    # train - validation split
    train_split = 0.95
    train_size = int(train_split * len(dataset))
    validn_size = len(dataset) - train_size
    # validn_size = 1024
    # train_size = len(dataset) - validn_size

    dataset_train, dataset_validn = torch.utils.data.random_split(
        dataset, [train_size, validn_size]
    )
    dataset_validn.transforms = transform_set["dataset_validn"]

    print(
        "Train set size = {}; Validation set size = {}".format(
            len(dataset_train), len(dataset_validn)
        )
    )

    # check if the random division does leads to very skewed distribution across classes
    # class0_count = sum([d[1] == 0 for d in dataset_validn])
    # class1_count = len(dataset_validn) - class0_count
    # print("Class-0 count = {}, class-1 count = {}".format(class0_count, class1_count))

    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_validn = DataLoader(dataset_validn, batch_size=1024)

    vgg_channel_list = [64, 128, 64]
    model = VggTypeNet(
        channel_list=vgg_channel_list, num_classes=1, batch_norm=batch_norm
    ).to(device)
    print(
        "Using vgg_channel_list = {}; Number of model parameters = {}".format(
            vgg_channel_list,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # optimizer = torch.optim.SGD(model.parameters())

    max_epoch = 1000
    best_validn_accuracy = -1
    for epoch in range(max_epoch):
        for idx_batch, batch in enumerate(dataloader_train):
            # print("Shape of the input: {}".format(batch[0].shape))
            # output = model.forward(batch[0])
            # print("Shape of the output: {}; \nsamples:\n{}".format(
            #     output.shape, output[0:2]))
            # break

            model.train()

            x, target = batch[0].to(device), batch[1].to(device)
            y = model(x).reshape(-1)

            train_loss = F.binary_cross_entropy_with_logits(
                input=y, target=target.float()
            )
            train_accuracy = (
                (torch.sigmoid(y) > 0.5) == target.byte()
            ).float().mean() * 100

            if idx_batch % 10 == 0:
                print(
                    "Epoch {}, batch {}: Training loss = {:5.3f}, accuracy = {:4.1f}".format(
                        epoch, idx_batch, train_loss, train_accuracy
                    )
                )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # update optimizer learning rate : multiply by 0.99 after every epoch
        if decreasing_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.95

        # ---------------------------------------------------------------------
        print("".join(["="] * 80))
        model.eval()

        validn_losses = []
        validn_accuracies = []
        for x_validn, target_validn in dataloader_validn:
            x_validn, target_validn = x_validn.to(device), target_validn.to(device)
            y_validn = None
            with torch.no_grad():
                y_validn = model(x_validn).reshape(-1)

            validn_losses.append(
                F.binary_cross_entropy_with_logits(
                    input=y_validn, target=target_validn.float()
                )
            )
            validn_accuracies.append(
                ((torch.sigmoid(y_validn) > 0.5) == target_validn.byte()).float()
            )

        validn_loss = sum(validn_losses) / len(validn_losses)
        validn_accuracy = torch.cat(validn_accuracies, dim=0).mean() * 100
        print(
            "Epoch {}: Validation ({} items) loss = {:5.3f}, accuracy = {:4.1f}".format(
                epoch, x_validn.shape[0], validn_loss, validn_accuracy
            )
        )

        if validn_accuracy > best_validn_accuracy:
            best_validn_accuracy = validn_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_path, "ep{}_accr{:4.1f}.pt".format(epoch, validn_accuracy)
                ),
            )

        print("".join(["="] * 80))
        # ---------------------------------------------------------------------


if __name__ == "__main__":
    # set seeds for reproducibility
    seed = 1
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    # add command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--train_folder",
        type=str,
        default="data/trainset/",
        help="Path to the folder having training images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/model/",
        help="Path to the folder where models to be saved",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_const",
        const=True,
        default=False,
        help="add this flag if you want to add batch_norm layer after every conv layer",
    )
    parser.add_argument(
        "--decreasing_lr",
        action="store_const",
        const=True,
        default=False,
        help="add this flag if you want to decrease learning-rate by multiplying with 0.99 after every epoch",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        args.train_folder,
        device=device,
        model_path=args.model_path,
        batch_norm=args.batch_norm,
        decreasing_lr=args.decreasing_lr,
    )

