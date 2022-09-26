"""
This script should be run to generate a dataset of undefective windows on the underlying undefective images.
"""
import torch
import torchvision
import pandas as pd
import numpy as np
import logging
import argparse
import os
import multiprocessing
import tqdm

from dataset_utils import has_blank_space


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


class UndefectiveWindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, train_df, image_dir, num_windows=100, window_size=160, yield_images=False, random_state=42):
        super(UndefectiveWindowDataset).__init__()

        self.image_dir = image_dir
        self.yield_images = yield_images
        self.num_windows = num_windows
        self.window_size = window_size
        self.random_state = random_state

        defective_imgs = set(train_df.ImageId.unique())
        all_imgs = [f for f in os.listdir(image_dir) if f.endswith('jpg')]
        self.undefective_imgs = [img for img in all_imgs if img not in defective_imgs]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "This dataset is only configured to work with multiprocess loading"
        per_worker = int(np.ceil(self.num_windows / worker_info.num_workers))
        wid = worker_info.id
        num_windows_for_this_worker = min(per_worker, self.num_windows - (1 + wid) * per_worker)
        worker_random_seed = 42 + wid  # Needed for numpy
        return self.image_generator(num_windows_for_this_worker, random_seed=worker_random_seed)

    def image_generator(self, num_windows, random_seed=42):
        gen = np.random.default_rng(random_seed)
        num_generated = 0
        while num_generated < num_windows:
            result = dict()
            index = gen.integers(len(self.undefective_imgs))
            image_id = self.undefective_imgs[index]
            img_filename = os.path.join(self.image_dir, image_id)
            log.debug(f'Loading image from {img_filename} and converting to grayscale')
            img = torchvision.io.read_image(img_filename, torchvision.io.ImageReadMode.GRAY)
            log.debug(f'img.shape={img.shape}')
            hw = self.window_size // 2
            extra = self.window_size % 2
            img_width = img.shape[2]
            window_center_min, window_center_max = hw, img_width - hw - extra
            window_center = gen.integers(window_center_min, window_center_max)
            min_x, max_x = int(window_center - hw), int(window_center + hw + extra)
            img = img[..., min_x:max_x]  # Tensor order is CHW

            if has_blank_space(img.numpy().transpose((2, 0, 1))):
                log.debug(f'Discarding window with blanks space')
                continue

            if self.yield_images:
                result['img'] = img

            result.update({
                'Index': index,
                'ImageId': image_id,
                'ClassId': 0,
                'instance_center_x': window_center,
                'window_size': self.window_size,
            })

            num_generated += 1
            yield result


if __name__ == "__main__":
    module_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--window_file',
        type=str,
        default=os.path.join(module_base, 'data', 'undefective_windows.csv'),
        help='File where generated windows will be saved.',
    )
    parser.add_argument(
        '--training_file',
        type=str,
        default=os.path.join(module_base, 'data', 'train.csv'),
        help='File to read defect segmentation masks from. (Used only to exclude these images.)',
    )
    parser.add_argument(
        '--num_windows',
        type=int,
        default=100,
        help='Generates this many windows.',
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=160,  # Magic value comes from defective dataset window size.
        help='Size of windows to generate.'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        help='Python logging module log level to use.',
    )
    parser.add_argument(
        '--dump_dir',
        type=str,
        default=None,
        help='Directory to dump generated windows, for debugging. If unset, does not dump.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for each pytorch dataset worker to yield'
    )

    args = parser.parse_args()
    log.setLevel(args.log_level)

    image_dir = os.path.join(module_base, 'data', 'train_images')
    print(f'Image directory is {image_dir}')

    log.info(f'Module base: {module_base}')
    log.info(f'Received args: {args}')

    yield_images = args.dump_dir is not None

    train_df = pd.read_csv(args.training_file)
    dataset = UndefectiveWindowDataset(
        train_df=train_df,
        image_dir=image_dir,
        num_windows=args.num_windows,
        window_size=args.window_size,
        yield_images=yield_images,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,  # Implies that yielded image tensors will be 4-dimensional (NCHW)
        num_workers=multiprocessing.cpu_count() - 1,
        prefetch_factor=2,
    )

    if yield_images:
        log.info(f'Dumping images to {args.dump_dir}')
        os.makedirs(args.dump_dir, exist_ok=True)

    log.info('Generating windows...')
    windows_df = pd.DataFrame()
    for batch in tqdm.tqdm(loader):
        if yield_images:
            for idx, img in zip(batch['Index'], batch['img']):  # It's batched, so iterate over the first dimension
                torchvision.io.write_png(img, os.path.join(args.dump_dir, str(idx.item()) + '.png'))
            del batch['img']
        # Why doesn't this dataframe need to be normalized? It's a pytorch mystery
        windows_df = pd.concat([windows_df, pd.DataFrame(data=batch)], ignore_index=True)
    log.info('Completed generating windows.')

    log.info(f'Writing output to {args.window_file}')
    keep_cols = [col for col in windows_df if col != 'img']
    windows_df.loc[:, keep_cols].to_csv(args.window_file)
