import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose

from model.models import VggTypeNet


def predict(
    image_folder="data/testset/",
    device=torch.device("cpu"),
    model_path="data/model/ep1_accr50.3.pt",
):

    assert os.path.exists(model_path)

    transforms = Compose([ToTensor()])
    dataset = ImageFolder(root=image_folder, transform=transforms)

    # answer = [(im_name[0], i) for i, im_name in enumerate(dataset.samples)]
    # print(answer)
    # return

    print("Number of images to predict on = {}".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=128)

    vgg_channel_list = [64, 128, 64]
    model = VggTypeNet(channel_list=vgg_channel_list, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data_count = 0
    answer = ["id,label\n"]  # header of submission file
    for i, (x, _) in enumerate(dataloader):

        print("predicting on batch {}".format(i + 1))
        x = x.to(device)

        y = model(x).reshape(-1)
        y = torch.sigmoid(y) > 0.5

        answer += [
            "{},{}\n".format(
                im_name[0].split("/")[-1].split(".")[0], "Dog" if y[i] else "Cat"
            )
            for i, im_name in enumerate(
                dataset.samples[data_count : data_count + x.shape[0]]
            )
        ]
        data_count += x.shape[0]

    submission_file = "submission.csv"
    with open(submission_file, "w+") as fl:
        fl.writelines(answer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/testset/",
        help="Path to the folder having images to predict on",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/model/ep1_accr50.3.pt",
        help="Path to the model file",
    )

    args = vars(parser.parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args["image_folder"], device=device, model_path=args["model_path"])
