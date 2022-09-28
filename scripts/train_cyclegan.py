"""
This script trains a CycleGAN model.
"""
import pandas as pd
import torch
import torchvision
import logging
import os

from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


class ImageWindowDataset(torch.utils.data.Dataset):
    def __init__(self, window_df: pd.DataFrame, image_dir, defect_classes=None, num_samples=None, random_state=42, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f'Using device: {device}')

        if defect_classes:
            log.info(f'Using only defect classes {defect_classes}')
            mask = window_df.ClassId.isin(defect_classes)
            window_df = window_df[mask]

        if num_samples:
            log.info(f'Taking {num_samples} samples')
            window_df = window_df.sample(n=num_samples, random_state=random_state)

        self.window_df = window_df
        self.image_dir = image_dir
        self.converter = torchvision.transforms.ConvertImageDtype(torch.float32)
        self.device = device

    def __getitem__(self, n):
        row = self.window_df.iloc[n, :]
        img = torchvision.io.read_image(os.path.join(self.image_dir, row.ImageId), torchvision.io.ImageReadMode.GRAY)
        hw = row.window_size // 2
        extra = row.window_size % 2
        x_min, x_max = int(row.instance_center_x - hw), int(row.instance_center_x + hw + extra)
        img = img[..., x_min:x_max]
        return self.converter(img)

    def __len__(self):
        return self.window_df.shape[0]


def train_model(x_gen, x_disc, y_gen, y_disc, x_domain, y_domain, n_epochs=10):
    """
    Jointly trains 4 networks (2 generators and 2 discriminators) that are cyclically
    linked to learn conditional mappings from an x domain to a y domain of images.

    :param x_gen:  Generator network from domain x to domain y.
    :param x_disc: Discriminator network for domain x. Indicates likelihood images are in domain x.
    :param y_gen:  Generator network from domain y to domain x.
    :param y_disc: Discriminator network for domain y. Indicates likelihood images are in domain y.
    :param x_domain: DataLoader yielding training images in the x domain.
    :param y_domain: DataLoader yielding training images in the y domain.
    """

    #  During each training step we will:
    #
    #       1. Minimize a "least squares loss" w.r.t. both discriminators' weights.
    #          For x_disc, this loss L := E[ (D_x(x) - 1)^2 ] + E[ D_x(G_y(y))^2 ]
    #          Where D_x is x_disc, and G_y is y_gen, and E[ ] indicates the expected
    #          value over a batch. y_disc is optimized analogously.
    #          (This loss is minimized when all "real" images in the x domain are
    #           assigned value 1, and when all "fake" images conditionally generated
    #           on images in the y domain are assigned value 0.)
    #
    #       2. Minimize another "least squares loss" w.r.t both generators' weights.
    #          For x_gen, this loss L := E[ D_y(G_x(x)) ] + lambda_f * E[ L1( G_y(G_x(x)) - x ) ] + lambda_b * E[ L1( G_x(G_y(y)) - y ) ]
    #                                    ^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                    Adversarial Loss             Forward cycle consistency loss         Backward cycle consistency loss
    #          Where the same notation for D_x, D_y, G_x, and G_y hold, and the cycle consistency losses are weighted
    #          with hyperparameters lambda_f and lambda_b, and we use an analagous loss term for y_gen.

    x_disc_loss_fn = torch.nn.MSELoss()
    x_disc_optimizer = torch.optim.Adam(x_disc.parameters(), lr=0.0002)
    fake_x_history = None  # A buffer to hold the history of fake x images.
    max_history_buffer_batches = 2
    tboard_summary_writer = SummaryWriter()
    global_step = 0
    for epoch_num in range(n_epochs):
        for i, (real_x_batch, real_y_batch) in enumerate(zip(x_domain, y_domain)):
            log.debug(f'epoch={epoch_num}, batch={i}')

            batch_size = real_x_batch.shape[0]
            fake_x_batch = y_gen(real_y_batch.to('cuda', non_blocking=True))

            if fake_x_history is None:
                fake_x_history = fake_x_batch.detach().to('cpu', non_blocking=True)

            num_historical_examples = fake_x_history.shape[0]

            #################################
            # Begin updating x_disc weights #
            #################################
            x_disc_optimizer.zero_grad()

            historical_fake_indices = torch.multinomial(input=torch.ones(num_historical_examples), num_samples=batch_size//2, replacement=False)
            historical_fake_x_batch = torch.index_select(input=fake_x_history, dim=0, index=historical_fake_indices)

            current_fake_indices = torch.multinomial(input=torch.ones(batch_size), num_samples=batch_size//2, replacement=False).to('cuda', non_blocking=True)
            current_fake_x_batch = torch.index_select(input=fake_x_batch, dim=0, index=current_fake_indices)

            real_labels = torch.ones(batch_size, 1, 1, 1).to('cuda', non_blocking=True)
            fake_labels = torch.zeros(batch_size//2, 1, 1, 1).to('cuda', non_blocking=True)

            real_loss = x_disc_loss_fn(x_disc(real_x_batch.to('cuda', non_blocking=True)), real_labels)
            fake_loss = x_disc_loss_fn(x_disc(historical_fake_x_batch.to('cuda', non_blocking=True)), fake_labels)
            fake_loss += x_disc_loss_fn(x_disc(current_fake_x_batch), fake_labels)
            loss = real_loss + fake_loss
            tboard_summary_writer.add_scalar('train/x_disc_loss', loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/x_disc_loss/real', real_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/x_disc_loss/fake', fake_loss.item(), global_step=global_step)

            loss.backward()
            x_disc_optimizer.step()

            #########################################
            # Update buffer of fake x domain images #
            #########################################
            num_saved_batches = num_historical_examples // batch_size
            if num_saved_batches < max_history_buffer_batches:
                torch.cat([fake_x_history, fake_x_batch.detach().to('cpu', non_blocking=True)], dim=0)
            elif num_saved_batches >= max_history_buffer_batches:
                replacement_batch_indices = torch.multinomial(input=torch.ones(num_historical_examples), num_samples=batch_size//2, replacement=False)
                detached = fake_x_batch.detach().to('cpu', non_blocking=True)
                for i, b in enumerate(replacement_batch_indices):
                    fake_x_history[b] = detached[i]

            global_step += 1

    return x_gen, x_disc, y_gen, y_disc
