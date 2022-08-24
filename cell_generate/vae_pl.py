import pytorch_lightning as pl
import torch
import torch.nn as nn
from .vae import Flatten


class VaePL(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        args,
        num_features=512,
        kld_weight=1,
        criterion=nn.MSELoss(),
        beta=4,
    ):
        super(VaePL, self).__init__()

        self.save_hyperparameters()
        self.num_features = num_features
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_type = str(type(encoder))[17:-2]
        self.decoder_type = str(type(decoder))[17:-2]

        self.args = args

        self.kld_weight = kld_weight
        self.criterion = criterion
        self.beta = beta

        self.flatten = Flatten()

        self.fc_mu = nn.Linear(512, self.num_features, bias=True)
        self.fc_var = nn.Linear(512, self.num_features, bias=True)
        self.deembedding = nn.Linear(self.num_features, 512)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate_autoencoder)

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
        inputs = batch[0]
        mu, log_var, feats = self._encode(inputs)
        z = self.reparametrize(mu, log_var)
        output = self._decode(z)

        loss, recon_loss, kld_loss = self.beta_loss(inputs, output, mu, log_var)

        self.log_dict(
            {
                "total": loss,
                "kl": kld_loss.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )

        return loss
