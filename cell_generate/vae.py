import torch
from torch import nn
from .encoders import generate_model
from .decoders import Decoder


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


class CVAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        num_features=512,
    ):
        super(CVAE, self).__init__()
        self.num_features = num_features
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_type = str(type(encoder))[17:-2]
        self.decoder_type = str(type(decoder))[17:-2]

        self.flatten = Flatten()

        self.fc_mu = nn.Linear(512, self.num_features, bias=True)
        self.fc_var = nn.Linear(512, self.num_features, bias=True)
        self.deembedding = nn.Linear(self.num_features, 512)

    def forward(self, x):

        mu, log_var, feats = self._encode(x)
        z = reparametrize(mu, log_var)
        output = self._decode(z)
        return output, mu, log_var, z, feats

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


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0)
        number of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out


if __name__ == "__main__":
    enc = generate_model(10)
    dec = Decoder()
    model = CVAE(encoder=enc, decoder=dec, num_features=128)
    inp = torch.rand((1, 2, 128, 128, 128))
    out = model(inp)
    print(out[0].shape)
    print(str(type(model))[17:-2])
