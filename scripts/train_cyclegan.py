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
import numpy as np
import argparse
import sys
import ignite

from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


class ImageWindowDataset(torch.utils.data.Dataset):
    def __init__(self, window_df: pd.DataFrame, image_dir, defect_classes=None, num_samples=None, random_state=42, device=None, read_mode=None):
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

        if read_mode and read_mode.lower() == 'gray':
            self.read_mode = torchvision.io.ImageReadMode.GRAY
        else:
            self.read_mode = torchvision.io.ImageReadMode.UNCHANGED
        log.info(f'Read mode is {self.read_mode}')

        self.window_df = window_df
        self.image_dir = image_dir

        test_row = self.window_df.iloc[0, :]
        test_img = torchvision.io.read_image(os.path.join(self.image_dir, test_row.ImageId), self.read_mode)
        hw = test_row.window_size // 2
        extra = test_row.window_size % 2
        x_min, x_max = int(test_row.instance_center_x - hw), int(test_row.instance_center_x + hw + extra)
        test_img = test_img[..., x_min:x_max]

        log.info(f'Image shapes: {test_img.shape}')
        n_channels = test_img.shape[0]
        means = stds = (0.5,) * n_channels
        embiggened_size = tuple(int(s * 1.12) for s in test_img.shape[1:])
        log.info(f'Embiggened size: {embiggened_size}')

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(embiggened_size, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(test_img.shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize(mean=means, std=stds),
        ])
        self.device = device

    def __getitem__(self, n):
        row = self.window_df.iloc[n, :]
        img = torchvision.io.read_image(os.path.join(self.image_dir, row.ImageId), self.read_mode)
        hw = row.window_size // 2
        extra = row.window_size % 2
        x_min, x_max = int(row.instance_center_x - hw), int(row.instance_center_x + hw + extra)
        img = img[..., x_min:x_max]
        return self.preprocess(img)

    def __len__(self):
        return self.window_df.shape[0]


def denorm_image(img: torch.Tensor):
    """
    De-normalizes an image, so that when displayed it appears like a natural scene instead of looking real glitchy.

    :param img: Normalized image.
    :return: de-normed image
    """
    n_channels = img.shape[0]
    device = img.device
    # These params would move a tanh output in the range [-1, 1] to [0, 1]
    scale = torch.Tensor([0.5] * n_channels).to(device)
    bias = torch.Tensor([0.5] * n_channels).to(device)
    scale = torch.unsqueeze(torch.unsqueeze(scale, 1), 2)
    bias = torch.unsqueeze(torch.unsqueeze(bias, 1), 2)
    with torch.no_grad():
        return img * scale + bias


def train_step(x_gen: torch.nn.Module, x_disc: torch.nn.Module, y_gen: torch.nn.Module, y_disc: torch.nn.Module,
               disc_optimizer: torch.optim.Optimizer, gen_optimizer: torch.optim.Optimizer,
               fake_x_history: torch.Tensor, fake_y_history: torch.Tensor,
               real_x_batch: torch.Tensor, real_y_batch: torch.Tensor,
               tboard_summary_writer: torch.utils.tensorboard.SummaryWriter,
               log_images: bool, global_step: int,
               max_history_buffer_batches=10):
    """
    Executes one training step.

    During each training step we will:

         1. Minimize a "least squares loss" w.r.t. both discriminators' weights.
            For x_disc, this loss L := E[ (D_x(x) - 1)^2 ] + E[ D_x(G_y(y))^2 ]
            Where D_x is x_disc, and G_y is y_gen, and E[ ] indicates the expected
            value over a batch. y_disc is optimized analogously.
            (This loss is minimized when all "real" images in the x domain are
             assigned value 1, and when all "fake" images conditionally generated
             on images in the y domain are assigned value 0.)

         2. Minimize another "least squares loss" w.r.t both generators' weights.
             L := E[ D_y(G_x(x)) ] + E[ D_x(G_y(y)) ] + lambda_f * E[ L1( G_y(G_x(x)) - x ) ] + lambda_b * E[ L1( G_x(G_y(y)) - y ) ]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^
                          Adversarial Losses                      Forward cycle consistency loss         Backward cycle consistency loss
            Where the same notation for D_x, D_y, G_x, and G_y hold, and the cycle consistency losses are weighted
            with hyperparameters lambda_f and lambda_b.

    :param x_gen:
    :param x_disc:
    :param y_gen:
    :param y_disc:
    :param disc_optimizer:
    :param gen_optimizer:
    :param fake_x_history:
    :param fake_y_history:
    :param real_x_batch:
    :param real_y_batch:
    :param tboard_summary_writer:
    :param log_images:
    :param global_step:
    :param max_history_buffer_batches:
    :return: x_gen, x_disc, y_gen, y_disc, disc_optimizer, gen_optimizer, fake_x_history, fake_y_history,
             all of which are mutated by this function
    """

    disc_loss_fn = torch.nn.MSELoss().to('cuda')
    cycle_loss_fn = torch.nn.L1Loss().to('cuda')
    identity_loss_fn = torch.nn.L1Loss().to('cuda')
    lambda_f = lambda_b = 10

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

    historical_fake_x_indices = torch.multinomial(input=torch.ones(num_historical_fake_x_examples), num_samples=max(x_batch_size // 2, 1),
                                                  replacement=True)
    historical_fake_x_batch = torch.index_select(input=fake_x_history, dim=0, index=historical_fake_x_indices)
    historical_fake_y_indices = torch.multinomial(input=torch.ones(num_historical_fake_y_examples), num_samples=max(y_batch_size // 2, 1),
                                                  replacement=True)
    historical_fake_y_batch = torch.index_select(input=fake_y_history, dim=0, index=historical_fake_y_indices)

    current_fake_x_indices = torch.multinomial(input=torch.ones(y_batch_size), num_samples=max(x_batch_size // 2, 1), replacement=True).to(
        'cuda', non_blocking=True)
    current_fake_x_batch = torch.index_select(input=fake_x_batch, dim=0, index=current_fake_x_indices)
    current_fake_y_indices = torch.multinomial(input=torch.ones(x_batch_size), num_samples=max(y_batch_size // 2, 1), replacement=True).to(
        'cuda', non_blocking=True)
    current_fake_y_batch = torch.index_select(input=fake_y_batch, dim=0, index=current_fake_y_indices)

    log.debug(f'real_x_batch_on_cuda.shape={real_x_batch_on_cuda.shape}')
    log.debug(f'historical_fake_x_batch.shape={historical_fake_x_batch.shape}')
    log.debug(f'historical_fake_y_batch.shape={historical_fake_y_batch.shape}')
    log.debug(f'real_y_batch_on_cuda.shape={real_y_batch_on_cuda.shape}')
    log.debug(f'current_fake_x_batch.shape={current_fake_x_batch.shape}')
    log.debug(f'current_fake_y_batch.shape={current_fake_y_batch.shape}')

    real_x_labels = torch.ones(x_batch_size, 1, 1, 1).to('cuda', non_blocking=True)
    fake_x_labels = torch.zeros(max(x_batch_size // 2, 1), 1, 1, 1).to('cuda', non_blocking=True)

    x_real_loss = disc_loss_fn(x_disc(real_x_batch_on_cuda), real_x_labels)
    x_fake_loss = disc_loss_fn(x_disc(historical_fake_x_batch.to('cuda', non_blocking=True)), fake_x_labels)
    x_fake_loss += disc_loss_fn(x_disc(current_fake_x_batch), fake_x_labels)
    x_disc_loss = x_real_loss + x_fake_loss
    tboard_summary_writer.add_scalar('train/x_disc_loss/real', x_real_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/x_disc_loss/fake', x_fake_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/x_disc_loss', x_disc_loss.item(), global_step=global_step)

    real_y_labels = torch.ones(y_batch_size, 1, 1, 1).to('cuda', non_blocking=True)
    fake_y_labels = torch.zeros(max(y_batch_size // 2, 1), 1, 1, 1).to('cuda', non_blocking=True)

    y_real_loss = disc_loss_fn(y_disc(real_y_batch_on_cuda), real_y_labels)
    y_fake_loss = disc_loss_fn(y_disc(historical_fake_y_batch.to('cuda', non_blocking=True)), fake_y_labels)
    y_fake_loss += disc_loss_fn(y_disc(current_fake_y_batch), fake_y_labels)
    y_disc_loss = y_real_loss + y_fake_loss
    tboard_summary_writer.add_scalar('train/y_disc_loss/real', y_real_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/y_disc_loss/fake', y_fake_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/y_disc_loss', y_disc_loss.item(), global_step=global_step)

    disc_loss = (x_disc_loss + y_disc_loss) / 2
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
                                                        num_samples=max(x_batch_size // 2, 1), replacement=True)
        fake_x_indices = torch.multinomial(input=torch.ones(fake_x_batch.shape[0]),
                                           num_samples=max(x_batch_size // 2, 1), replacement=True)
        replacement_batch_y_indices = torch.multinomial(input=torch.ones(num_historical_fake_y_examples),
                                                        num_samples=max(y_batch_size // 2, 1), replacement=True)
        fake_y_indices = torch.multinomial(input=torch.ones(fake_y_batch.shape[0]),
                                           num_samples=max(y_batch_size // 2, 1), replacement=True)
        detached_x = fake_x_batch.detach().to('cpu', non_blocking=True)
        detached_y = fake_y_batch.detach().to('cpu', non_blocking=True)
        for b, i in zip(replacement_batch_x_indices, fake_x_indices):
            fake_x_history[b] = detached_x[i]
        for b, i in zip(replacement_batch_y_indices, fake_y_indices):
            fake_y_history[b] = detached_y[i]

    ############################
    # Update generator weights #
    ############################
    gen_optimizer.zero_grad()

    # These have already been used to optimize the discriminator, and pytorch doesn't like it when you re-use tensors
    # for two optimizers. So although it's inefficient, we regenerate these.
    fake_x_batch = y_gen(real_y_batch_on_cuda)
    fake_y_batch = x_gen(real_x_batch_on_cuda)
    recovered_x = y_gen(fake_y_batch)
    recovered_y = x_gen(fake_x_batch)

    if log_images:
        tboard_summary_writer.add_image('train/images/real_x', denorm_image(real_x_batch[0]), global_step=global_step)
        tboard_summary_writer.add_image('train/images/fake_x', denorm_image(fake_x_batch[0]), global_step=global_step)
        tboard_summary_writer.add_image('train/images/real_y', denorm_image(real_y_batch[0]), global_step=global_step)
        tboard_summary_writer.add_image('train/images/fake_y', denorm_image(fake_y_batch[0]), global_step=global_step)
        tboard_summary_writer.add_image('train/images/recovered_x', denorm_image(recovered_x[0]), global_step=global_step)
        tboard_summary_writer.add_image('train/images/recovered_y', denorm_image(recovered_y[0]), global_step=global_step)

    x_adversarial_loss = disc_loss_fn(x_disc(fake_x_batch), real_y_labels)
    y_adversarial_loss = disc_loss_fn(y_disc(fake_y_batch), real_x_labels)
    tboard_summary_writer.add_scalar('train/x_adversarial_loss', x_adversarial_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/y_adversarial_loss', y_adversarial_loss.item(), global_step=global_step)

    x_domain_cycle_loss = cycle_loss_fn(
        real_x_batch_on_cuda,
        recovered_x
    )
    y_domain_cycle_loss = cycle_loss_fn(
        real_y_batch_on_cuda,
        recovered_y
    )
    tboard_summary_writer.add_scalar('train/x_domain_cycle_loss', x_domain_cycle_loss.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/y_domain_cycle_loss', y_domain_cycle_loss.item(), global_step=global_step)

    identity_loss_x = identity_loss_fn(x_gen(real_y_batch_on_cuda), real_y_batch_on_cuda) * lambda_f / 2
    identity_loss_y = identity_loss_fn(y_gen(real_x_batch_on_cuda), real_x_batch_on_cuda) * lambda_b / 2
    tboard_summary_writer.add_scalar('train/identity_loss_x', identity_loss_x.item(), global_step=global_step)
    tboard_summary_writer.add_scalar('train/identity_loss_y', identity_loss_y.item(), global_step=global_step)

    generator_loss = x_adversarial_loss + y_adversarial_loss + lambda_f * x_domain_cycle_loss + lambda_b * y_domain_cycle_loss + \
                     identity_loss_x + identity_loss_y
    tboard_summary_writer.add_scalar('train/generator_loss', generator_loss.item(), global_step=global_step)

    generator_loss.backward()
    gen_optimizer.step()

    return x_gen, x_disc, y_gen, y_disc, disc_optimizer, gen_optimizer, fake_x_history, fake_y_history


def train_model(x_gen: torch.nn.Module, x_disc: torch.nn.Module, y_gen: torch.nn.Module, y_disc: torch.nn.Module,
                x_domain: torch.utils.data.DataLoader, y_domain: torch.utils.data.DataLoader,
                tboard_summary_writer: torch.utils.tensorboard.SummaryWriter,
                n_epochs: int = 10, model_save_base_path: str = 'models'):
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

    os.makedirs(model_save_base_path, exist_ok=True)

    disc_optimizer = torch.optim.Adam(itertools.chain(x_disc.parameters(), y_disc.parameters()), lr=0.0002, betas=[0.5, 0.999])
    disc_constant_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(disc_optimizer, lr_lambda=lambda x: 1)
    disc_linear_scheduler = torch.optim.lr_scheduler.LinearLR(disc_optimizer, start_factor=1, end_factor=0, total_iters=100)
    # Scheduler switches from no decay to linear decay at epoch 100
    disc_scheduler = torch.optim.lr_scheduler.SequentialLR(disc_optimizer, schedulers=[disc_constant_scheduler, disc_linear_scheduler],
                                                           milestones=[100])

    gen_optimizer = torch.optim.Adam(itertools.chain(y_gen.parameters(), x_gen.parameters()), lr=0.0002, betas=[0.5, 0.999])
    gen_constant_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(gen_optimizer, lr_lambda=lambda x: 1)
    gen_linear_scheduler = torch.optim.lr_scheduler.LinearLR(gen_optimizer, start_factor=1, end_factor=0, total_iters=100)
    # Scheduler switches from no decay to linear decay at epoch 100
    gen_scheduler = torch.optim.lr_scheduler.SequentialLR(gen_optimizer, schedulers=[gen_constant_scheduler, gen_linear_scheduler],
                                                          milestones=[100])

    fake_x_history = None  # A buffer to hold the history of fake x images.
    fake_y_history = None  # A buffer to hold the history of fake x images.

    global_step = 0
    fid_fn = ignite.metrics.FID(device='cuda')
    for epoch_num in range(n_epochs):
        fid_fn.reset()

        for batch_num, (real_x_batch, real_y_batch) in enumerate(zip(x_domain, y_domain)):
            log.debug(f'epoch={epoch_num}, batch={batch_num}')
            log_images = batch_num % 50 == 0
            x_gen, x_disc, y_gen, y_disc, disc_optimizer, gen_optimizer, fake_x_history, fake_y_history = \
                train_step(x_gen, x_disc, y_gen, y_disc, disc_optimizer, gen_optimizer, fake_x_history, fake_y_history,
                           real_x_batch, real_y_batch, tboard_summary_writer, log_images, global_step)
            real_y_batch_on_cuda = real_y_batch.detach().to('cuda')
            real_x_batch_on_cuda = real_x_batch.detach().to('cuda')
            fid_eval_size = min(real_x_batch.shape[0], real_y_batch.shape[0])
            fid_fn.update((y_gen(real_y_batch_on_cuda[:fid_eval_size]), real_x_batch_on_cuda[:fid_eval_size]))
            fid_fn.update((y_gen(real_x_batch_on_cuda[:fid_eval_size]), real_y_batch_on_cuda[:fid_eval_size]))
            global_step += 1

        disc_scheduler.step()
        gen_scheduler.step()
        tboard_summary_writer.add_scalar('train/disc_scheduler_lr', disc_scheduler.get_last_lr()[0], global_step=global_step)
        tboard_summary_writer.add_scalar('train/gen_scheduler_lr', gen_scheduler.get_last_lr()[0], global_step=global_step)

        tboard_summary_writer.add_scalar('train/fid', fid_fn.compute(), global_step=global_step)

        torch.save(x_gen.state_dict(), os.path.join(model_save_base_path, f'x_gen.{epoch_num}.pth'))
        torch.save(x_disc.state_dict(), os.path.join(model_save_base_path, f'x_disc.{epoch_num}.pth'))
        torch.save(y_gen.state_dict(), os.path.join(model_save_base_path, f'y_gen.{epoch_num}.pth'))
        torch.save(y_disc.state_dict(), os.path.join(model_save_base_path, f'y_disc.{epoch_num}.pth'))

    return x_gen, x_disc, y_gen, y_disc


if __name__ == '__main__':
    try:
        module_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script_path = os.path.join(module_base, 'scripts')
    except NameError:
        # __file__ not defined
        # On CDSW be sure to set this environment variable to point to the dir containing the project scripts
        script_path = os.environ['SCRIPTS_PATH']
        module_base = os.path.dirname(script_path)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Number of epochs to run training.',
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        help='Python logging module log level to use.',
    )
    parser.add_argument(
        '--model_save_dir',
        type=str,
        default='models',
        help='Directory to save learned models.'
    )
    parser.add_argument(
        '--tboard_log_dir',
        type=str,
        default=None,
        help='Directory to save tensorboard logs.'
    )
    parser.add_argument(
        '--x_class',
        type=int,
        action='append',
        default=None,
        help='Specify defect classes for the x domain. Leave blank to use all.'
   )
    parser.add_argument(
        '--x_image_base',
        type=str,
        default=os.path.join(module_base, 'data', 'train_images'),
        help='The base image path for the x domain, to which the ClassId will be appended to form the path to the image.'
    )
    parser.add_argument(
        '--y_image_base',
        type=str,
        default=os.path.join(module_base, 'data', 'train_images'),
        help='The base image path for the y domain, to which the ClassId will be appended to form the path to the image.'
    )
    parser.add_argument(
        '--x_windows',
        type=str,
        default=os.path.join(module_base, 'data', 'windows.csv'),
        help='File to specify subregions of images to use for the x domain.'
    )
    parser.add_argument(
        '--y_windows',
        type=str,
        default=os.path.join(module_base, 'data', 'undefective_windows.csv'),
        help='File to specify subregions of images to use for the y domain.'
    )
    parser.add_argument(
        '--read_grayscale',
        action='store_true',
        default=False,
        help='Used to force reading in grayscale mode for images. Leave unset to read unchanged.',
    )
    parser.add_argument(
        '--x_disc_model',
        type=str,
        default=None,
        help='PatchGANDiscriminator model weights for x domain to be loaded from disk.',
    )
    parser.add_argument(
        '--y_disc_model',
        type=str,
        default=None,
        help='PatchGANDiscriminator model weights for y domain to be loaded from disk.',
    )
    parser.add_argument(
        '--x_gen_model',
        type=str,
        default=None,
        help='GeneratorModel model weights for x domain to be loaded from disk.',
    )
    parser.add_argument(
        '--y_gen_model',
        type=str,
        default=None,
        help='GeneratorModel model weights for y domain to be loaded from disk.',
    )

    args = parser.parse_args()
    log.setLevel(args.log_level)

    if script_path not in sys.path:
        sys.path.append(script_path)

    from cyclegan_models import GeneratorModel, PatchGANDiscriminator

    log.info(f'Module base: {module_base}')
    log.info(f'Received args: {args}')

    read_mode = None
    if args.read_grayscale:
        read_mode = 'gray'

    df = pd.read_csv(args.x_windows, index_col=0)
    defective_images = ImageWindowDataset(df, defect_classes=args.x_class, image_dir=args.x_image_base, read_mode=read_mode)
    log.info(f'Length of x domain dataset: {len(defective_images)}')

    df = pd.read_csv(args.y_windows, index_col=0)
    undefective_images = ImageWindowDataset(df, image_dir=args.y_image_base, read_mode=read_mode)
    log.info(f'Length of y domain dataset: {len(undefective_images)}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f'Using device {device}')

    num_batches = 100
    x_batch_size = int(np.ceil(len(defective_images) / num_batches))
    y_batch_size = int(np.ceil(len(undefective_images) / num_batches))
    log.info(f'Batch sizes: x_batch_size={x_batch_size} and y_batch_size={y_batch_size}')

    num_x_channels = defective_images[0].shape[0]
    num_y_channels = defective_images[0].shape[0]
    log.info(f'X channels: {num_x_channels}, Y channels: {num_y_channels}')
    assert num_x_channels == num_y_channels, f'Image domains have different numbers of channels, which is not yet supported.' \
                                             f' (xch={num_x_channels}, ych={num_y_channels})'

    def init_weights(module: torch.nn.Module):
        if type(module) == torch.nn.Conv2d:
            log.info(f'Initializing conv weights')
            torch.nn.init.normal_(module.weight, mean=0, std=.02)

    x_disc = PatchGANDiscriminator(cin=num_x_channels).to(device)
    if args.x_disc_model:
        log.info(f'Loading x_disc_model from {args.x_disc_model}')
        x_disc.load_state_dict(torch.load(args.x_disc_model))
    else:
        x_disc.apply(init_weights)

    x_gen = GeneratorModel(cin=num_x_channels).to(device)
    if args.x_gen_model:
        log.info(f'Loading x_gen_model from {args.x_gen_model}')
        x_gen.load_state_dict(torch.load(args.x_gen_model))
    else:
        x_gen.apply(init_weights)

    x_domain = torch.utils.data.DataLoader(defective_images, batch_size=x_batch_size, pin_memory=True, shuffle=True)

    y_disc = PatchGANDiscriminator(cin=num_y_channels).to(device)
    if args.y_disc_model:
        log.info(f'Loading y_disc_model from {args.y_disc_model}')
        y_disc.load_state_dict(torch.load(args.y_disc_model))
    else:
        y_disc.apply(init_weights)

    y_gen = GeneratorModel(cin=num_y_channels).to(device)
    if args.y_gen_model:
        log.info(f'Loading y_gen_model from {args.y_gen_model}')
        y_gen.load_state_dict(torch.load(args.y_gen_model))
    else:
        y_gen.apply(init_weights)

    y_domain = torch.utils.data.DataLoader(undefective_images, batch_size=y_batch_size, pin_memory=True, shuffle=True)

    tboard_summary_writer = SummaryWriter(log_dir=args.tboard_log_dir)

    train_model(x_gen, x_disc, y_gen, y_disc, x_domain, y_domain,
                tboard_summary_writer=tboard_summary_writer,
                n_epochs=args.num_epochs, model_save_base_path=args.model_save_dir)