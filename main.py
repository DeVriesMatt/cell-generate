import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import torchio as tio
import argparse

from cell_generate.reporting import get_experiment_name

from cell_generate.vae import CVAE
from cell_generate.training_functions import train
from cell_generate.encoders import generate_model
from cell_generate.decoders import Decoder
from cell_generate.datasets import SingleCell


def train_vae(args):
    enc = generate_model(args.depth)
    dec = Decoder()
    autoencoder = CVAE(encoder=enc, decoder=dec, num_features=args.num_features)
    everything_working = True
    file_not_found = False
    wrong_architecture = False
    try:
        checkpoint = torch.load(args.pretrained_path)
    except FileNotFoundError:
        print(
            "This model doesn't exist."
            " Please check the provided path and try again. "
            "Ignore this message if you do not have a pretrained model."
        )
        checkpoint = {"model_state_dict": None}
        file_not_found = True
        everything_working = False
    except AttributeError:
        print("No pretrained model given.")
        checkpoint = {"model_state_dict": None}
        everything_working = False
    except:
        print("No pretrained model given.")
        checkpoint = {"model_state_dict": None}
        everything_working = False
    try:
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
        print(f"The loss of the loaded model is {checkpoint['loss']}")
    except RuntimeError:
        print("The model architecture given doesn't " "match the one provided.")
        print("Training from scratch")
        wrong_architecture = True
        everything_working = False
    except AttributeError or TypeError:
        print("Training from scratch")
    except:
        print("Training from scratch")

    spatial_transforms = {
        tio.RandomElasticDeformation(): 0.2,
        tio.RandomAffine(): 0.8,
    }

    transform = tio.Compose(
        [
            tio.CropOrPad((args.image_size, args.image_size, args.image_size)),
            tio.OneOf(spatial_transforms, p=0.5),
            tio.RandomFlip(axes=["LR", "AP", "IS"]),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ]
    )

    dataset = SingleCell(
        image_path=args.dataset_path,
        dataframe_path=args.dataframe_path,
        transforms=transform,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    reconstruction_criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=args.learning_rate_autoencoder * 16 / args.batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    logging_info = get_experiment_name(model=autoencoder, output_dir=args.output_dir)
    name_logging, name_model, name_writer, name_images, name = logging_info
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)

    if everything_working:
        logging.info(
            f"Started training cluster model {name} at {now} "
            f"using autoencoder which is "
            f"saved at {args.pretrained_path}."
        )
        print(
            f"Started training model {name} at {now}."
            f"using autoencoder which is s"
            f"aved at {args.pretrained_path}."
        )
    if file_not_found:
        logging.info(
            f"The autoencoder model at {args.pretrained_path}"
            f" doesn't exist."
            f"if you knew this already, then don't worry. "
            f"If not, then check the path and try again"
        )
        logging.info("Training from scratch")
        print(
            f"The autoencoder model at "
            f"{args.pretrained_path} doesn't exist. "
            f"If you knew this already, then don't worry. "
            f"If not, then check the path and try again"
        )
        print("Training from scratch")

    if wrong_architecture:
        logging.info(
            f"The autoencoder model at {args.pretrained_path} has "
            f"a different architecture to the one provided "
            f"If not, then check the path and try again"
        )
        logging.info("Training from scratch")
        print(
            f"The autoencoder model at {args.pretrained_path} "
            f"has a different architecture to the one provided "
            f"If not, then check the path and try again."
        )
        print("Training from scratch")

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")
        print(f"Argument {arg}: {value}")

    autoencoder, name_logging, name_model, name_writer, name = train(
        model=autoencoder,
        dataloader=dataloader,
        num_epochs=args.num_epochs_autoencoder,
        criterion=reconstruction_criterion,
        optimizer=optimizer,
        logging_info=logging_info,
        kld_weight=args.kld_weight,
        beta=args.beta,
    )
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"Finished training at {now}.")
    print(f"Finished training at {now}.")

    return autoencoder, name_logging, name_model, name_writer, name


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Single cell generational models")

    parser.add_argument(
        "--dataset_path",
        default="/home/mvries/Documents/Datasets/" "OPM/SingleCellFromNathan_17122021/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--dataframe_path",
        default="/home/mvries/Documents/Datasets/OPM/"
        "SingleCellFromNathan_17122021/all_data_removedwrong_ori_removedTwo.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/CVAEOutput/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of " "features to extract.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.00001,
        type=float,
        help="Please provide the learning rate " "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI"
        "/DYNCESYS/mvries/Projects/TearingNetNew/Reconstruct_dgcnn_cls_k20_plane/models/shapenetcorev2_250.pkl",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=250,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--beta",
        default=4,
        type=int,
        help="Please provide a value for beta for the beta-vae"
        ". See https://openreview.net/forum?id=Sy2fzU9gl.",
    )
    parser.add_argument(
        "--kld_weight",
        default=1,
        type=int,
        help="Please provide a value for Kullback_liebler convergence weight"
        ". See https://openreview.net/forum?id=Sy2fzU9gl.",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--depth",
        default=10,
        type=int,
        help="ResNet depth.",
    )
    parser.add_argument(
        "--image_size",
        default=128,
        type=int,
        help="Input image size.",
    )
    args = parser.parse_args()
    output = train_vae(args)
