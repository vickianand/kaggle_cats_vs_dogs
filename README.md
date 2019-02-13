# Cats and Dogs classification - Kaggle competition for IFT6135H19 Assignment 1

## Data

Download data from the [Kaggle competition page](https://www.kaggle.com/c/ift6135h19/data).

## Training

Run `python train.py --train_folder <path to folder having images>`

## Prediction

Run

```bash
python predict.py --image_folder <path to folder having images> --model_path <path to model state_dict file>
```

Note : `<path to folder having images>` Should have images in subfolders and not directly inside the folder. Subfolders are considered are classes.
For prediction a single dummy subfolder can be created. In this case class is obviously ignored.
