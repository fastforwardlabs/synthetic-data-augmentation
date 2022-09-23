"""
This script should be run to create a regularized version of the dataset that is suitable
for classification and synthetic data augmentation tasks.
"""
import torch
import torchvision
import pandas as pd
import numpy as np
import logging
import argparse
import os
import multiprocessing


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


def rle_to_indices(rle_string, imsize=(256, 1600)):
    """
    Converts run-length encoded pairs into pairs of indices.
    Returns a tuple of (array of x values, array of y values)

    :param rle_string: Run length encoded string indicating segmentation mask
    :param imsize: Image size used to decode RLE string
    :return: Tuple consisting of (array of x points, array of y points)
    """
    nums = [int(x) for x in rle_string.split()]
    N = len(nums)
    xs = np.asarray([])
    ys = np.asarray([])
    for i in range(N//2):
        n, length = nums[2*i:2*i+2] # Are the pixels in the dataset 1-indexed, instead of 0-indexed?
        ns = [n - 1 + i for i in range(length)]
        ys_, xs_ = np.unravel_index(ns, shape=imsize, order='F')
        xs = np.append(xs, xs_)
        ys = np.append(ys, ys_)
    return xs.astype(int), ys.astype(int)


def get_adjacents(x, y):
    """
    Given a point (x, y), returns a list of adjacent points.

    :param x: X value
    :param y: Y value
    :return: List of (x, y) tuples
    """
    deltas = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    )
    return [(x + dx, y + dy) for (dx, dy) in deltas]


def identify_instances(indices):
    """
    Given a tuple of (x indices, y_indices) indicating defect masks,
    partitions the set of points into contiguous instances using a
    flood fill algorithm.

    :param indices: Tuple of arrays of indices.
    :return: A list of tuples of length N, where N is the number of instances.
             Each tuple contains the (x indices, y indices) pair for an instance.
    """
    result = []
    unidentified = set((x, y) for (x, y) in zip(*indices))
    while unidentified:
        instance = set()
        points_to_visit = {next(iter(unidentified))}
        while points_to_visit:
            for point in points_to_visit.copy():
                instance.add(point)
                unidentified.remove(point)
                for neighbor in get_adjacents(*point):
                    if neighbor in unidentified:
                        points_to_visit.add(neighbor)
            points_to_visit = points_to_visit.difference(instance)
        result.append(np.asarray(list(instance)))
    return result


def get_overlapping_instances(row, df, window_size=256):
    """
    Given a particular row, a dataframe, and a window size, computes the instances that
    would appear in a window centered around the row's instance with a certain width.
    Only considers the x-axis. Intended to be used with DataFrame.apply

    :param row: row from df to analyze
    :param df: dataframe of all instances
    :param window_size: Window size to consider for what's overlapping
    :return: df's indices of overlapping instances
    """
    hw = window_size // 2
    extra = window_size % 2
    window_min, window_max = row.instance_center_x - hw, row.instance_center_x + hw + extra
    df = df[df.ImageId == row.ImageId]
    mask = (window_min < df.instance_min_x) & (df.instance_min_x < window_max)
    mask |= (window_min < df.instance_max_x) & (df.instance_max_x < window_max)
    mask |= (df.instance_min_x < window_min) & (window_max < df.instance_max_x)
    return df[mask].index


def all_instances_fit_in_window(row, df, max_size=145):
    """
    Given a row and a dataframe of instances, determines whether the previously computed overlapping instances
    are all within `max_size` distance of the row's instance center.

    :param row: row from df to analyze
    :param df: dataframe of all instances
    :param max_size: Maximum allowable size of instances
    :return: True or False, in regard to the row's overlapping instances
    """
    overlapping = df.loc[row.overlapping_instances]
    return np.all(overlapping.instance_size <= max_size)


def all_instances_approx_centered(row, df, tol=10):
    """
    Given a row and a dataframe of instances, determines whether the previously computed overlapping instances
    are all within `tol` distance of the row's instance center.
    """
    overlapping = df.loc[row.overlapping_instances]
    appx_center_min = row.instance_center_x - tol
    appx_center_max = row.instance_center_x + tol
    return np.all((appx_center_min < overlapping.instance_center_x) &
                 (overlapping.instance_center_x < appx_center_max))


def overlaps_edge(row, img_width=1600):
    """
    Determines whether a row in an instance dataframe overlaps the edge of the image
    :param row: Row from instance dataframe
    :param img_width: Image width
    :return: True or False
    """
    hw = row.window_size // 2
    extra = row.window_size % 2
    min_x, max_x = row.instance_center_x - hw, row.instance_center_x + hw + extra
    return (min_x < 0) or (max_x > img_width)


def has_blank_space(img, tol=25):
    """
    The heuristic for detecting blank space is if one of the image columns is close to zero.
    If the mean of all pixel vals is less than tol, we say it's close to zero.

    :param img: A grayscale image with the x-dimension in the first index.
    :param tol: A column's mean pixel values should be below this level.
    :return:
    """
    return np.any([np.mean(col) < tol for col in img])


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, train_df, defect_classes=(1, 3)):
        num_classes_by_image = train_df.groupby(by='ImageId').size()
        multiple_defect_classes = num_classes_by_image[num_classes_by_image > 1]
        multiple_defect_classes_mask = train_df.ImageId.isin(multiple_defect_classes.index)

        log.info(f'Removing {np.sum(multiple_defect_classes_mask)} of {train_df.shape[0]} images with multiple defect classes')
        self.df = train_df[(~multiple_defect_classes_mask) & train_df.ClassId.isin(defect_classes)]
        self.defect_classes = defect_classes

    def __getitem__(self, n):
        index = self.df.index[n]
        indices = rle_to_indices(self.df.at[index, 'EncodedPixels'])
        instances = identify_instances(indices)
        result = []
        for instance in instances:
            result.append({
                'ImageId': self.df.at[index, 'ImageId'],
                'ClassId': self.df.at[index, 'ClassId'],
                'instance_center_x': np.mean(instance[:, 0]),
                'instance_min_x': np.min(instance[:, 0]),
                'instance_max_x': np.max(instance[:, 0]),
                'instance_std_x': np.std(instance[:, 0]),
            })
        return result

    def __len__(self):
        return self.df.shape[0]


class RegularizationDataset(torch.utils.data.IterableDataset):
    def __init__(self, instances_df, image_dir, window_size=None, instance_size_margin=5, instance_center_tol=42, yield_images=False):
        super(RegularizationDataset).__init__()

        instances_df['instance_size'] = instances_df.instance_max_x - instances_df.instance_min_x
        quantile_levels = [.9, .95, .99]
        size_quantiles = instances_df.instance_size.quantile(quantile_levels)
        log.info(f'Instance size quantiles (quantile_levels={quantile_levels}): {size_quantiles.values}')
        if not window_size:
            window_size = size_quantiles.loc[.90]

        log.info(f'Using window_size={window_size}')
        instances_df['window_size'] = window_size

        self.image_dir = image_dir
        self.yield_images = yield_images
        self.instances_df = instances_df
        self.instance_size_margin = instance_size_margin
        self.instance_center_tol = instance_center_tol
        self.window_size = window_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "This dataset is only configured to work with multiprocess loading"
        per_worker = int(np.ceil(self.instances_df.shape[0] / worker_info.num_workers))
        wid = worker_info.id
        iter_start = wid * per_worker
        iter_end = min(iter_start + per_worker, self.instances_df.shape[0])
        return self.image_generator(iter_start, iter_end)

    def image_generator(self, start, end):
        indices = self.instances_df.index[start:end]
        my_instances = self.instances_df.loc[indices, :].copy()

        log.debug(f'For worker {torch.utils.data.get_worker_info().id}, '
                  f'Getting overlapping instances with window_size={self.window_size}')
        my_instances['overlapping_instances'] = my_instances.apply(
            lambda row: get_overlapping_instances(row, self.instances_df, window_size=self.window_size), axis=1)

        max_instance_size = self.window_size - self.instance_size_margin
        log.debug(f'For worker {torch.utils.data.get_worker_info().id}, '
                  f'Determining which windows have well-fitted instances, with max_instance_size={max_instance_size}')
        my_instances['all_instances_fit_in_window'] = my_instances.apply(
            lambda row: all_instances_fit_in_window(row, self.instances_df, max_size=max_instance_size),
            axis=1
        )

        log.debug(f'For worker {torch.utils.data.get_worker_info().id}, '
                  f'Determining which windows have approximately centered instances, with '
                  f'instance_center_tol={self.instance_center_tol}')
        my_instances['all_overlapping_appx_centered'] = my_instances.apply(
            lambda row: all_instances_approx_centered(row, self.instances_df, tol=self.instance_center_tol),
            axis=1
        )

        mask = my_instances.all_overlapping_appx_centered & my_instances.all_instances_fit_in_window
        log.debug(f'For worker {torch.utils.data.get_worker_info().id}, Discarding {np.sum(mask)} of '
                  f'{my_instances.shape[0]} windows that are not well-centered.')
        my_instances = my_instances[mask]

        for row in my_instances.itertuples():
            if overlaps_edge(row):
                log.debug('Discarding window that overlaps edge.')
                continue

            image_path = os.path.join(self.image_dir, row.ImageId)
            log.debug(f'Loading image from {image_path} and converting to grayscale')
            img = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
            log.debug(f'img.shape={img.shape}')
            hw = row.window_size // 2
            extra = row.window_size % 2
            min_x, max_x = int(row.instance_center_x - hw), int(row.instance_center_x + hw + extra)
            img = img[..., min_x:max_x]  # Tensor order is CHW
            log.debug(f'window img.shape={img.shape}')

            if has_blank_space(img.numpy().transpose((2, 0, 1))):  # Transpose the x-dimension to the first position
                log.debug('Discarding window with blank space.')
                continue

            row_dict = row._asdict()
            del row_dict['overlapping_instances']  # Not returnable if compatibility with DataLoader is desired
            if self.yield_images:
                row_dict['img'] = img
            yield row_dict


def normalize_df_cols(df):
    for col in df:
        if col in ('ImageId', 'img'):
            pass
        else:
            df[col] = df[col].apply(lambda x: x.item())
    return df


if __name__ == "__main__":
    module_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--instance_id',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Boolean flag indicating whether to run instance identification step',
    )
    parser.add_argument(
        '--window_generation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Boolean flag indicating whether to run window generation step',
    )
    parser.add_argument(
        '--instance_file',
        type=str,
        default=os.path.join(module_base, 'data', 'instances.csv'),
        help='File where identified instances will be saved/loaded',
    )
    parser.add_argument(
        '--window_file',
        type=str,
        default=os.path.join(module_base, 'data', 'windows.csv'),
        help='File where generated windows will be saved.',
    )
    parser.add_argument(
        '--training_file',
        type=str,
        default=os.path.join(module_base, 'data', 'train.csv'),
        help='File to read defect segmentation masks from',
    )
    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Take only the first N examples from the training file. Set to 0 to use all.',
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

    args = parser.parse_args()
    log.setLevel(args.log_level)

    log.info(f'Module base: {module_base}')
    log.info(f'Received args: {args}')
    instance_df = None # This object is conditionally created

    if args.instance_id:
        log.info('Running instance identification')
        train_df = pd.read_csv(args.training_file)
        if args.head:
            log.info(f'Taking first {args.head} elements.')
            train_df = train_df.head(args.head)
        dataset = InstanceDataset(train_df)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=multiprocessing.cpu_count() - 1,
            prefetch_factor=2,
        )
        log.info('Identifying instances...')
        data = list(loader)
        log.info('Completed instance identification.')

        flat_data = [i for instances in data for i in instances]
        instance_df = pd.DataFrame(data=flat_data)
        log.info(f'Writing output to {args.instance_file}')

        instance_df = normalize_df_cols(instance_df)
        instance_df.to_csv(args.instance_file)

    if args.window_generation:
        log.info('Generating windows from instances')

        if instance_df is None:
            log.info(f'Loading instance info from {args.instance_file}')
            instance_df = pd.read_csv(args.instance_file, index_col=0)
        else:
            log.info(f'Using in-memory instance info')

        image_dir = os.path.join(module_base, 'data', 'train_images')
        log.info(f'Image directory: {image_dir}')
        yield_images = args.dump_dir is not None
        dataset = RegularizationDataset(instance_df, image_dir, yield_images=yield_images)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Implies that yielded image tensors will be 3-dimensional (CHW) and not 4-dimensional
            num_workers=multiprocessing.cpu_count() - 1,
            prefetch_factor=2,
        )
        log.info('Generating regularized windows...')
        rows = list(loader)
        log.info('Completed generating regularized windows.')

        windows_df = pd.DataFrame(data=rows)  # Why doesn't this dataframe need to be normalized? It's a pytorch mystery

        log.info(f'Writing output to {args.window_file}')
        keep_cols = [col for col in windows_df if col != 'img']
        windows_df.loc[:, keep_cols].to_csv(args.window_file)

        if yield_images:
            log.info(f'Dumping images to {args.dump_dir}')
            os.makedirs(args.dump_dir, exist_ok=True)
            windows_df.apply(lambda row: torchvision.io.write_png(
                row.img,
                os.path.join(args.dump_dir, str(row.Index) + '.png')
            ), axis=1)

