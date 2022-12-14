import pytorch_lightning as pl
import torch
import torch.nn as nn
import tifffile
import os

from .vae import Flatten


class VaePL(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        args,
        save_images_path=None,
        num_features=512,
        kld_weight=1,
        criterion=nn.MSELoss(),
        beta=4,
    ):
        super(VaePL, self).__init__()
        self.save_hyperparameters(ignore=['criterion', 'encoder', 'decoder'])
        self.num_features = num_features
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_type = str(type(encoder))[17:-2]
        self.decoder_type = str(type(decoder))[17:-2]

        self.args = args
        self.lr = args.learning_rate_autoencoder
        self.save_images_path = save_images_path

        self.kld_weight = kld_weight
        self.criterion = criterion
        self.beta = beta

        self.flatten = Flatten()

        self.fc_mu = nn.Linear(512, self.num_features, bias=True)
        self.fc_var = nn.Linear(512, self.num_features, bias=True)
        self.deembedding = nn.Linear(self.num_features, 512)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr
        )

    def beta_loss(self, inputs, outputs, mu, log_var):
        recon_loss = self.criterion(inputs, outputs)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recon_loss + self.beta * self.kld_weight * kld_loss

        return loss, recon_loss, kld_loss

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps

    def _encode(self, x):
        feats = self.encoder(x)
        x = self.flatten(feats)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var, feats

    def _decode(self, z):
        z = self.deembedding(z)
        z = z.unsqueeze(1)
        output = self.decoder(z)
        return output

    def training_step(self, batch, batch_idx):
        log_dir_for_images = self.trainer.log_dir
        try:
            log_dir_for_images = log_dir_for_images + "/images/"
            os.makedirs(log_dir_for_images, exist_ok=True)
        except:
            print("Trainer log dir returning None type")

        inputs = batch[0]
        mu, log_var, feats = self._encode(inputs)
        z = self.reparametrize(mu, log_var)
        output = self._decode(z)

        loss, recon_loss, kld_loss = self.beta_loss(inputs, output, mu, log_var)

        if (batch_idx % 10 == 0) and (self.save_images_path is not None):
            try:
                tifffile.imwrite(
                    log_dir_for_images
                    + f"/input_{self.current_epoch}_{str(batch_idx).zfill(5)}.tif",
                    inputs[0].detach().cpu().numpy(),
                )
                tifffile.imwrite(
                    log_dir_for_images
                    + f"/output_{self.current_epoch}_{str(batch_idx).zfill(5)}.tif",
                    output[0].detach().cpu().numpy(),
                )
            except:
                print("can't save images")

        self.log_dict(
            {
                "total": loss,
                "kl": kld_loss.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )
        self.log("loss", loss, prog_bar=True)

        return loss
