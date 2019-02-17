import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import transforms

from model.models import TwoConvOnePool, VggTypeNet

import warnings

# warnings.filterwarnings("ignore")


def get_validn_mispreds(
    train_folder, device, model_path="model/m5_ep267_ac94.4.pt", batch_norm=False
):

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

    dataset = ImageFolder(root=train_folder, transform=transform_set["dataset_validn"])
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

    print(
        "Train set size = {}; Validation set size = {}".format(
            len(dataset_train), len(dataset_validn)
        )
    )

    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_validn = DataLoader(dataset_validn, batch_size=1024)

    vgg_channel_list = [64, 128, 64]
    model = VggTypeNet(
        channel_list=vgg_channel_list, num_classes=1, batch_norm=batch_norm
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(
        "Using vgg_channel_list = {}; Number of model parameters = {}".format(
            vgg_channel_list,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    print("".join(["="] * 80))
    model.eval()

    clear_mispreds = []
    close_mispreds = []
    cat_path = "visualize/cat_mispreds/"
    dog_path = "visualize/dog_mispreds/"
    boundary_path = "visualize/boundary_mispreds/"
    os.makedirs(cat_path, exist_ok=True)
    os.makedirs(dog_path, exist_ok=True)
    os.makedirs(boundary_path, exist_ok=True)

    validn_losses = []
    validn_accuracies = []
    for x_validn, target_validn in dataloader_validn:
        x_validn, target_validn = x_validn.to(device), target_validn.to(device)
        y_validn = None
        with torch.no_grad():
            y_validn = model(x_validn).reshape(-1)

        # get clearly mis-predicted images
        sigmoid_val = torch.sigmoid(y_validn)
        clear_miss_cats_bool = ((sigmoid_val > 0.5) != target_validn.byte()) * (
            sigmoid_val > 0.9985
        )
        clear_miss_dogs_bool = ((sigmoid_val > 0.5) != target_validn.byte()) * (
            sigmoid_val < 0.02
        )
        cat_4grid = []
        dog_4grid = []
        for i, im in enumerate(x_validn[clear_miss_cats_bool]):
            save_image(im, filename=cat_path + "{}.jpg".format(i), nrow=1)
            if i < 4:
                cat_4grid.append(im)
        for i, im in enumerate(x_validn[clear_miss_dogs_bool]):
            save_image(im, filename=dog_path + "{}.jpg".format(i), nrow=1)
            if i < 4:
                dog_4grid.append(im)

        # get boundary cases
        boundary_4grid = []
        boundary_cases_bool = (sigmoid_val > 0.47) * (sigmoid_val < 0.53)
        for i, im in enumerate(x_validn[boundary_cases_bool]):
            save_image(im, filename=boundary_path + "{}.jpg".format(i), nrow=1)
            if i < 4:
                boundary_4grid.append(im)

        print(
            "found {} cats mispred, {} dogs mispreds and {} boundary cases".format(
                clear_miss_cats_bool.sum(),
                clear_miss_dogs_bool.sum(),
                boundary_cases_bool.sum(),
            )
        )
        save_image(cat_4grid, filename=cat_path + "cat_4grid.jpg", nrow=2)
        save_image(dog_4grid, filename=dog_path + "dog_4grid.jpg", nrow=2)
        save_image(boundary_4grid, filename=boundary_path + "boundry_4grid.jpg", nrow=2)

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
    # print(
    #     "Validation ({} items) loss = {:5.3f}, accuracy = {:4.1f}".format(
    #         x_validn.shape[0], validn_loss, validn_accuracy
    #     )
    # )

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
        default="model/m5_ep267_ac94.4.pt",
        help="Path to the model which is to used for prediction",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_const",
        const=True,
        default=False,
        help="add this flag if you want to add batch_norm layer after every conv layer",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_validn_mispreds(
        args.train_folder,
        device=device,
        model_path=args.model_path,
        batch_norm=args.batch_norm,
    )

