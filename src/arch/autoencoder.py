import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yaml

from src.utils.constants import SETUP_FILE

class Autoencoder(nn.Module):
    def __init__(self, input_dim, arch_name):
        super(Autoencoder, self).__init__()

        with open(SETUP_FILE, "r") as yaml_file:
            self.arch_info = yaml.safe_load(yaml_file)[arch_name]

        mlsize = self.arch_info["hidden_dim"]

        mlsize.insert(0, input_dim)

        encoder_layers = []
        for i in range(1, len(mlsize)):
            encoder_layers.append(nn.Linear(mlsize[i-1], mlsize[i]))
            

            if i < len(mlsize) - 1:
                encoder_layers.append(nn.ReLU())

                if self.arch_info["norm"] is True:
                    decoder_layers.append(nn.BatchNorm1d(mlsize[i-2]))
            
                encoder_layers.append(nn.Dropout(self.arch_info["dropout"]))
        
        encoder_layers.append(nn.ReLU())
        
        latent_layers = []
        if self.arch_info["norm"] is True:
            latent_layers.append(nn.BatchNorm1d(mlsize[-1]))
        latent_layers.append(nn.Dropout(self.arch_info["dropout"]))

        decoder_layers = []
        for i in range(len(mlsize), 1, -1):
            decoder_layers.append(nn.Linear(mlsize[i-1], mlsize[i-2]))
        
            if i > 2:
                decoder_layers.append(nn.ReLU())

                decoder_layers.append(nn.Dropout(self.arch_info["dropout"]))
            
        decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)
        self.latent = nn.Sequential(*latent_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x

    def reset_parameters(self):

        self.encoder.reset_parameters()
        self.latent.reset_parameters()
        self.decoder.reset_parameters()
    
    def scaler(self):
        upper_limit = 1
        lower_limit = 0

        return MinMaxScaler(feature_range=(lower_limit, upper_limit))