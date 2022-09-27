"""
Defines models used for training CycleGAN for synthetic data generation.
"""
import torch
import torchvision
import logging


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_features):
        # Architecture according to "Unpaired Image-to-Image Translation
        # using Cycle-Consistent Adversarial Networks," by Zhu et al.
        super(ResidualBlock, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=0, bias=False),
            torch.nn.InstanceNorm2d(num_features=num_features),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=0, bias=False),
            torch.nn.InstanceNorm2d(num_features=num_features),
        )

    def forward(self, input):
        residual = self.model(input)
        crop = torchvision.transforms.CenterCrop(residual.shape[2:])
        return residual + crop(input)


class GeneratorModel(torch.nn.Module):
    def __init__(self, cin):
        # Architecture according to "Unpaired Image-to-Image Translation
        # using Cycle-Consistent Adversarial Networks," by Zhu et al.
        super(GeneratorModel, self).__init__()

        self.model = torch.nn.Sequential(
            torchvision.transforms.Pad(padding=40, padding_mode='reflect'),

            torch.nn.Conv2d(in_channels=cin, out_channels=32, kernel_size=9, stride=1, padding=4, bias=False),
            torch.nn.InstanceNorm2d(num_features=32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(num_features=64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(num_features=128),
            torch.nn.ReLU(),

            ResidualBlock(num_features=128),
            ResidualBlock(num_features=128),
            ResidualBlock(num_features=128),
            ResidualBlock(num_features=128),
            ResidualBlock(num_features=128),

            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                     output_padding=1, bias=False),
            torch.nn.InstanceNorm2d(num_features=64),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                     output_padding=1, bias=False),
            torch.nn.InstanceNorm2d(num_features=32),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(in_channels=32, out_channels=cin, kernel_size=9, stride=1, padding=4,
                                     output_padding=0, bias=False),
            torch.nn.InstanceNorm2d(num_features=cin),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        log.debug(f'Input shape={input.shape}')
        return self.model(input)
