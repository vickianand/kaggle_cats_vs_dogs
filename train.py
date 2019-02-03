import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose

from model.models import TwoConvOnePool, VggTypeNet

import warnings

# warnings.filterwarnings("ignore")


def main():

    transforms = Compose([ToTensor()])

    dataset = ImageFolder(root="data/trainset/", transform=transforms)
    # dataset = torch.utils.data.Subset(dataset, indices=range(2000))

    # train - validation split
    train_split = 0.95
    train_size = int(train_split * len(dataset))
    validn_size = len(dataset) - train_size
    # validn_size = 1024
    # train_size = len(dataset) - validn_size

    dataset_train, dataset_validn = torch.utils.data.random_split(
        dataset, [train_size, validn_size]
    )

    print("Train set size = {}; Validation set size = {}".format(
        len(dataset_train), len(dataset_validn)))

    # # check if the random division does leads to very skewed distribution across classes
    # class0_count = sum([d[1] == 0 for d in dataset_validn])
    # class1_count = len(dataset_validn) - class0_count
    # print("Class-0 count = {}, class-1 count = {}".format(class0_count, class1_count))

    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_validn = DataLoader(dataset_validn, batch_size=1024)

    vgg_channel_list = [64, 128, 64]
    model = VggTypeNet(channel_list=vgg_channel_list, num_classes=1)
    print("Using vgg_channel_list = {}; Number of model parameters = {}".format(
        vgg_channel_list, sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    max_epoch = 1000
    for epoch in range(max_epoch):
        for idx_batch, batch in enumerate(dataloader_train):
            # print("Shape of the input: {}".format(batch[0].shape))
            # output = model.forward(batch[0])
            # print("Shape of the output: {}; \nsamples:\n{}".format(
            #     output.shape, output[0:2]))
            # break

            model.train()

            x, target = batch
            y = model(x).reshape(-1,)

            train_loss = F.binary_cross_entropy_with_logits(
                input=y, target=target.float())
            train_accuracy = ((torch.sigmoid(y) > 0.5) ==
                              target.byte()).float().mean() * 100

            print("Epoch {}, batch {}: Training loss = {}, accuracy = {}".format(
                epoch, idx_batch, train_loss, train_accuracy))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # ---------------------------------------------------------------------
        print(''.join(['=']*80))
        model.eval()

        x_validn, target_validn = next(iter(dataloader_validn))
        y_validn = model(x_validn).reshape(-1,)

        validn_loss = F.binary_cross_entropy_with_logits(
            input=y_validn, target=target_validn.float())
        validn_accuracy = ((torch.sigmoid(y_validn) > 0.5)
                           == target_validn.byte()).float().mean() * 100

        print("Epoch {}: Validation ({} items) loss = {}, accuracy = {}".format(
            epoch, x_validn.shape[0], validn_loss, validn_accuracy))

        print(''.join(['=']*80))
        # ---------------------------------------------------------------------


if __name__ == "__main__":
    main()
