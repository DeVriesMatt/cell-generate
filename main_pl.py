import torch
from torch.utils.data import DataLoader
import torchio as tio
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cell_generate.vae_pl import VaePL
from cell_generate.encoders import generate_model
from cell_generate.decoders import Decoder
from cell_generate.datasets import SingleCell
# testing

def train_vae_pl(args):
    enc = generate_model(args.depth)
    dec = Decoder()
    save_images_path = args.output_dir + "/images/"
    find_lr = True
    autoencoder = VaePL(
        encoder=enc,
        decoder=dec,
        num_features=args.num_features,
        args=args,
        save_images_path=save_images_path,
    )

    try:
        # checkpoint = torch.load(args.pretrained_path, map_location="c")
        # print(checkpoint["hyper_parameters"])
        autoencoder.load_from_checkpoint(args.pretrained_path)
    except Exception as e:
        print(f"Can't load pretrained network due to error {e}")

    spatial_transforms = {
        tio.RandomElasticDeformation(): 0.2,
        tio.RandomAffine(): 0.8,
    }

    transform = tio.Compose(
        [
            tio.CropOrPad((args.image_size, args.image_size, args.image_size)),
            tio.OneOf(spatial_transforms, p=0.8),
            tio.RandomFlip(axes=["LR", "AP", "IS"]),
        ]
    )

    dataset = SingleCell(
        image_path=args.dataset_path,
        dataframe_path=args.dataframe_path,
        transforms=transform,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir, save_top_k=1, monitor="loss"
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,
        precision=16,
        max_epochs=args.num_epochs_autoencoder,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback],
    )

    if find_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(autoencoder, dataloader, num_training=100)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig(args.output_dir + "lr_finder_plot.png")

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(new_lr)

        # update hparams of the model
        autoencoder.hparams.lr = new_lr

    trainer.fit(autoencoder, dataloader)


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
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/"
        "mvries/ResultsAlma/cell-generate/logs/lightning_logs/version_10233842/checkpoints/"
        "epoch=3-step=4096.ckpt",
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
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="The number of gpus to use for training.",
    )
    args = parser.parse_args()
    train_vae_pl(args)
