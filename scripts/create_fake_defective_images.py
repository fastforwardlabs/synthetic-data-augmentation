"""
This script creates fake defective images using a trained CycleGAN model.
"""
import pandas as pd
import torch.utils.data
import torchvision
import logging
import os
import argparse

from tqdm import tqdm

from datasets import ImageWindowDataset, denorm_image
from cyclegan_models import GeneratorModel


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


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
        '--log_level',
        type=str,
        default='INFO',
        help='Python logging module log level to use.',
    )
    parser.add_argument(
        '--image_base',
        type=str,
        default=os.path.join(module_base, 'data', 'train_images'),
        help='The base image path for undefective images, to which the ClassId will be appended to form the path to the image.'
    )
    parser.add_argument(
        '--undefective_csv',
        type=str,
        default=os.path.join(module_base, 'data', 'train_undefective_only.csv'),
        help='File to specify subregions of images to use for undefective samples.'
    )
    parser.add_argument(
        '--read_grayscale',
        action='store_true',
        default=False,
        help='Used to force reading in grayscale mode for images. Leave unset to read unchanged.',
    )
    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default=None,
        help='Model weights to be loaded from disk.',
        required=True,
    )
    parser.add_argument(
        '--num_dataloader_threads_per_dataset',
        type=int,
        default=1,
        help='Number of threads to use for torch.utils.data.DataLoader per dataset'
             ' (this amount will be used per dataloader).'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Takes only the first N batches from undefective dataloader.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(module_base, 'data', 'synthetic_images'),
        help='The base image path for synthetic images, to which the ClassId will be appended to form the path to the image.'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=os.path.join(module_base, 'data', 'synthetic_images.csv'),
        help='CSV file where synthetic image crops will be written.'
    )
    parser.add_argument(
        '--class_id',
        type=str,
        default='1',
        help='The fake ClassId to be used when writing out the crops file. It cannot be inferred so must be supplied.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Dataloader batch size.',
    )
    parser.add_argument(
        '--save_undefective',
        action='store_true',
        default=False,
        help='Save undefective images, for debugging.',
    )

    args = parser.parse_args()
    log.setLevel(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f'Using device: {device}')

    log.info(f'Module base: {module_base}')
    log.info(f'Received args: {args}')

    read_mode = None
    num_channels = 3
    if args.read_grayscale:
        read_mode = 'gray'
        num_channels = 1
    log.info(f'Num channels: {num_channels}')

    df = pd.read_csv(args.undefective_csv, index_col=0)
    undefective_set = ImageWindowDataset(df, image_dir=args.image_base, read_mode=read_mode)
    log.info(f'Length of undefective_set: {len(undefective_set)}')

    undefective_loader = torch.utils.data.DataLoader(undefective_set, batch_size=args.batch_size, pin_memory=True,
                                                     num_workers=args.num_dataloader_threads_per_dataset)

    generator = GeneratorModel(cin=num_channels).to(device)
    log.info(f'Loading generator model from {args.pretrained_weights}')
    generator.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    output_data = {
        'ImageId': [],
        'ClassId': [],
        'instance_center_x': [],
        'window_size': [],
    }

    with torch.inference_mode():
        for batchnum, batch in tqdm(enumerate(undefective_loader)):
            if args.head and batchnum >= args.head:
                log.info(f'Stopping after {args.head} batches')
                break
            batch = batch.to(device)
            fake_batch = generator(batch)
            for j, image in enumerate(fake_batch):
                num_channels = image.shape[0]
                if num_channels == 1:
                    image = image.expand(3, -1, -1)
                image_id = f'{batchnum}_{j}.jpg'
                image_path = os.path.join(args.output_dir, image_id)
                torchvision.utils.save_image(denorm_image(image), image_path)
                output_data['ImageId'].append(image_id)
                output_data['ClassId'].append(args.class_id)
                output_data['instance_center_x'].append(image.shape[2] // 2)
                output_data['window_size'].append(image.shape[2])
                if args.save_undefective:
                    undefective_image_path = os.path.join(args.output_dir, f'{batchnum}_{j}.undefective.jpg')
                    torchvision.utils.save_image(denorm_image(batch[j].expand(3, -1, -1)), undefective_image_path)

    output_df = pd.DataFrame(data=output_data)
    output_df.to_csv(args.output_csv)
