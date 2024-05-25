from torch import nn
from torch import optim
import lightning as pl

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        return z, h

class DeepGuardPredictor(pl.LightningModule):
    def __init__(self, vocab_size, input_dim, model_dim, output_dim, lr, warmup, max_iters, dropout=0.0, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = AutoEncoder(self.hparams.input_dim, self.hparams.model_dim, self.hparams.output_dim, self.hparams.dropout)
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError