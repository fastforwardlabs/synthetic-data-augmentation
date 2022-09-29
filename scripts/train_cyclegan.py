"""
This script trains a CycleGAN model.
"""
import pandas as pd
import torch
import torch.utils.data
import torchvision
import logging
import os
import itertools


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


def train_model(x_gen: torch.nn.Module, x_disc: torch.nn.Module, y_gen: torch.nn.Module, y_disc: torch.nn.Module,
                x_domain: torch.utils.data.DataLoader, y_domain: torch.utils.data.DataLoader, n_epochs=10):
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
    #          with hyperparameters lambda_f and lambda_b, and we use an analogous loss term for y_gen.

    disc_loss_fn = torch.nn.MSELoss()
    cycle_loss_fn = torch.nn.L1Loss()
    disc_optimizer = torch.optim.Adam(itertools.chain(x_disc.parameters(), y_disc.parameters()), lr=0.0002)
    gen_optimizer = torch.optim.Adam(itertools.chain(y_gen.parameters(), x_gen.parameters()), lr=0.0002)
    fake_x_history = None  # A buffer to hold the history of fake x images.
    fake_y_history = None  # A buffer to hold the history of fake x images.
    max_history_buffer_batches = 10
    tboard_summary_writer = SummaryWriter()
    global_step = 0
    lambda_f = lambda_b = 10
    for epoch_num in range(n_epochs):
        for i, (real_x_batch, real_y_batch) in enumerate(zip(x_domain, y_domain)):
            log.debug(f'epoch={epoch_num}, batch={i}')

            x_batch_size = real_x_batch.shape[0]
            y_batch_size = real_y_batch.shape[0]
            log.debug(f'x_batch_size={x_batch_size}, y_batch_size={y_batch_size}')

            real_x_batch_on_cuda = real_x_batch.to('cuda', non_blocking=True)
            real_y_batch_on_cuda = real_y_batch.to('cuda', non_blocking=True)
            fake_x_batch = y_gen(real_y_batch_on_cuda)
            fake_y_batch = x_gen(real_x_batch_on_cuda)

            if fake_x_history is None:
                fake_x_history = fake_x_batch.detach().to('cpu', non_blocking=True)

            if fake_y_history is None:
                fake_y_history = fake_y_batch.detach().to('cpu', non_blocking=True)

            num_historical_fake_x_examples = fake_x_history.shape[0]
            num_historical_fake_y_examples = fake_y_history.shape[0]
            log.debug(f'num_historical_y_examples={num_historical_fake_y_examples}')

            ################################
            # Update discriminator weights #
            ################################
            disc_optimizer.zero_grad()

            historical_fake_x_indices = torch.multinomial(input=torch.ones(num_historical_fake_x_examples), num_samples=max(y_batch_size//2, 1),
                                                          replacement=True)
            historical_fake_x_batch = torch.index_select(input=fake_x_history, dim=0, index=historical_fake_x_indices)
            historical_fake_y_indices = torch.multinomial(input=torch.ones(num_historical_fake_y_examples), num_samples=max(x_batch_size//2, 1),
                                                          replacement=True)
            historical_fake_y_batch = torch.index_select(input=fake_y_history, dim=0, index=historical_fake_y_indices)

            current_fake_x_indices = torch.multinomial(input=torch.ones(y_batch_size), num_samples=max(y_batch_size//2, 1), replacement=False).to('cuda', non_blocking=True)
            current_fake_x_batch = torch.index_select(input=fake_x_batch, dim=0, index=current_fake_x_indices)
            current_fake_y_indices = torch.multinomial(input=torch.ones(x_batch_size), num_samples=max(x_batch_size//2, 1), replacement=False).to('cuda', non_blocking=True)
            current_fake_y_batch = torch.index_select(input=fake_y_batch, dim=0, index=current_fake_y_indices)

            log.debug(f'real_x_batch_on_cuda.shape={real_x_batch_on_cuda.shape}')
            log.debug(f'historical_fake_x_batch.shape={historical_fake_x_batch.shape}')
            log.debug(f'historical_fake_y_batch.shape={historical_fake_y_batch.shape}')
            log.debug(f'real_y_batch_on_cuda.shape={real_y_batch_on_cuda.shape}')
            log.debug(f'current_fake_x_batch.shape={current_fake_x_batch.shape}')
            log.debug(f'current_fake_y_batch.shape={current_fake_y_batch.shape}')

            real_x_labels = torch.ones(x_batch_size, 1, 1, 1).to('cuda', non_blocking=True)
            fake_x_labels = torch.zeros(max(y_batch_size//2, 1), 1, 1, 1).to('cuda', non_blocking=True)

            x_real_loss = disc_loss_fn(x_disc(real_x_batch_on_cuda), real_x_labels)
            x_fake_loss = disc_loss_fn(x_disc(historical_fake_x_batch.to('cuda', non_blocking=True)), fake_x_labels)
            x_fake_loss += disc_loss_fn(x_disc(current_fake_x_batch), fake_x_labels)
            x_disc_loss = x_real_loss + x_fake_loss
            tboard_summary_writer.add_scalar('train/x_disc_loss/real', x_real_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/x_disc_loss/fake', x_fake_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/x_disc_loss', x_disc_loss.item(), global_step=global_step)

            real_y_labels = torch.ones(y_batch_size, 1, 1, 1).to('cuda', non_blocking=True)
            fake_y_labels = torch.zeros(max(x_batch_size//2, 1), 1, 1, 1).to('cuda', non_blocking=True)

            y_real_loss = disc_loss_fn(y_disc(real_y_batch_on_cuda), real_y_labels)
            y_fake_loss = disc_loss_fn(y_disc(historical_fake_y_batch.to('cuda', non_blocking=True)), fake_y_labels)
            y_fake_loss += disc_loss_fn(y_disc(current_fake_y_batch), fake_y_labels)
            y_disc_loss = y_real_loss + y_fake_loss
            tboard_summary_writer.add_scalar('train/y_disc_loss/real', y_real_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/y_disc_loss/fake', y_fake_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/y_disc_loss', y_disc_loss.item(), global_step=global_step)

            disc_loss = x_disc_loss + y_disc_loss
            tboard_summary_writer.add_scalar('train/disc_loss', disc_loss.item(), global_step=global_step)
            disc_loss.backward()
            disc_optimizer.step()

            ###############################################
            # Update buffer of fake x and y domain images #
            ##############################################
            num_saved_batches = num_historical_fake_x_examples // y_batch_size
            log.debug(f'num_historical_fake_x_examples={num_historical_fake_x_examples}')
            log.debug(f'y_batch_size={y_batch_size}')
            log.debug(f'num_historical_fake_y_examples={num_historical_fake_y_examples}')
            log.debug(f'x_batch_size={x_batch_size}')
            log.debug(f'num_saved_batches={num_saved_batches}')
            log.debug(f'max_history_buffer_batches={max_history_buffer_batches}')
            if num_saved_batches < max_history_buffer_batches:
                fake_x_history = torch.cat([fake_x_history, fake_x_batch.detach().to('cpu', non_blocking=True)], dim=0)
                fake_y_history = torch.cat([fake_y_history, fake_y_batch.detach().to('cpu', non_blocking=True)], dim=0)
            elif num_saved_batches >= max_history_buffer_batches:
                replacement_batch_x_indices = torch.multinomial(input=torch.ones(num_historical_fake_x_examples),
                                                                num_samples=max(y_batch_size//2, 1), replacement=False)
                replacement_batch_y_indices = torch.multinomial(input=torch.ones(num_historical_fake_y_examples),
                                                                num_samples=max(x_batch_size//2, 1), replacement=False)
                detached_x = fake_x_batch.detach().to('cpu', non_blocking=True)
                detached_y = fake_y_batch.detach().to('cpu', non_blocking=True)
                for i, b in enumerate(replacement_batch_x_indices):
                    fake_x_history[b] = detached_x[i]
                for i, b in enumerate(replacement_batch_y_indices):
                    fake_y_history[b] = detached_y[i]

            ############################
            # Update generator weights #
            ############################
            gen_optimizer.zero_grad()

            # These have already been used to optimize the discriminator, and pytorch doesn't like it when you re-use tensors
            # for two optimizers. So although it's inefficient, we regenerate these.
            fake_x_batch = y_gen(real_y_batch_on_cuda)
            fake_y_batch = x_gen(real_x_batch_on_cuda)

            x_adversarial_loss = disc_loss_fn(x_disc(fake_x_batch), real_y_labels)
            y_adversarial_loss = disc_loss_fn(y_disc(fake_y_batch), real_x_labels)
            tboard_summary_writer.add_scalar('train/x_adversarial_loss', x_adversarial_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/y_adversarial_loss', y_adversarial_loss.item(), global_step=global_step)

            x_domain_cycle_loss = cycle_loss_fn(
                real_x_batch_on_cuda,
                y_gen(fake_y_batch)
            )
            y_domain_cycle_loss = cycle_loss_fn(
                real_y_batch_on_cuda,
                x_gen(fake_x_batch)
            )
            tboard_summary_writer.add_scalar('train/x_domain_cycle_loss', x_domain_cycle_loss.item(), global_step=global_step)
            tboard_summary_writer.add_scalar('train/y_domain_cycle_loss', y_domain_cycle_loss.item(), global_step=global_step)

            generator_loss = x_adversarial_loss + y_adversarial_loss + lambda_f * x_domain_cycle_loss + lambda_b * y_domain_cycle_loss
            tboard_summary_writer.add_scalar('train/generator_loss', generator_loss.item(), global_step=global_step)

            generator_loss.backward()
            gen_optimizer.step()

            global_step += 1

    return x_gen, x_disc, y_gen, y_disc
