"""
This script trains a classifier model.
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

from datasets import LabeledImageWindowDataset


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
        '--image_base',
        type=str,
        default=os.path.join(module_base, 'data', 'train_images'),
        help='The base image path for the x domain, to which the ClassId will be appended to form the path to the image.'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default=os.path.join(module_base, 'data', 'train.csv'),
        help='File to specify subregions of images to use for training split.'
    )
    parser.add_argument(
        '--val_csv',
        type=str,
        default=os.path.join(module_base, 'data', 'val.csv'),
        help='File to specify subregions of images to use for validation split.'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default=os.path.join(module_base, 'data', 'test.csv'),
        help='File to specify subregions of images to use for test split.'
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
    )
    parser.add_argument(
        '--num_dataloader_threads_per_dataset',
        type=int,
        default=1,
        help='Number of threads to use for torch.utils.data.DataLoader per dataset'
             ' (this amount will be used per dataloader).'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Number of epochs to continue training while validation loss has not improved.'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Takes only the first N examples from each dataset. 0 uses all.'
    )
    parser.add_argument(
        '--oversample_minority_class',
        action='store_true',
        default=False,
        help='Oversamples minority class to reach same data length as majority class.'
    )
    parser.add_argument(
        '--synthetic_csv',
        type=str,
        default=None,
        help='File to specify synthetic images to augment training set.'
    )
    parser.add_argument(
        '--synthetic_image_base',
        type=str,
        default=os.path.join(module_base, 'data', 'synthetic_images'),
        help='Base directory for synthetic images.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate for Adam optimizer'
    )

    args = parser.parse_args()
    log.setLevel(args.log_level)

    log.info(f'Module base: {module_base}')
    log.info(f'Received args: {args}')

    read_mode = None
    if args.read_grayscale:
        read_mode = 'gray'

    df = pd.read_csv(args.train_csv, index_col=0)
    labels = df.ClassId.unique()
    num_labels = len(labels)
    remapped_labels = {label: i for (i, label) in enumerate(labels)}
    log.info(f'Mapping labels to ordinals (label: ordinal): {remapped_labels}')
    df.ClassId = df.ClassId.apply(lambda x: remapped_labels[x])
    if args.head:
        df = df.iloc[:args.head]

    if args.oversample_minority_class:
        class_counts = df.ClassId.value_counts().sort_values()  # smallest first
        majority_class = class_counts.index[-1]
        minority_class = class_counts.index[0]
        num_majority_class = class_counts.at[majority_class]
        num_minority_class = class_counts.at[minority_class]
        log.info(f'num_majority_class={num_majority_class}, num_minority_class={num_minority_class}')
        num_to_sample = num_majority_class - num_minority_class
        oversampled = df[df.ClassId == minority_class].sample(n=num_to_sample, replace=True, random_state=42)
        df = pd.concat([df, oversampled], ignore_index=True)

    df['image_base'] = args.image_base

    if args.synthetic_csv:
        synthetic_df = pd.read_csv(args.synthetic_csv, index_col=0)
        log.info(f'Num synthetic examples: {synthetic_df.shape[0]}')
        synthetic_df['image_base'] = args.synthetic_image_base
        df = pd.concat([df, synthetic_df], ignore_index=True)

    train_set = LabeledImageWindowDataset(df, read_mode=read_mode)
    log.info(f'Length of train dataset: {len(train_set)}')

    df = pd.read_csv(args.val_csv, index_col=0)
    df.ClassId = df.ClassId.apply(lambda x: remapped_labels[x])
    df['image_base'] = args.image_base
    if args.head:
        df = df.iloc[:args.head]
    val_set = LabeledImageWindowDataset(df, read_mode=read_mode)
    log.info(f'Length of val dataset: {len(val_set)}')

    df = pd.read_csv(args.test_csv, index_col=0)
    df.ClassId = df.ClassId.apply(lambda x: remapped_labels[x])
    df['image_base'] = args.image_base
    if args.head:
        df = df.iloc[:args.head]
    test_set = LabeledImageWindowDataset(df, read_mode=read_mode)
    log.info(f'Length of test dataset: {len(test_set)}')

    def init_weights(module: torch.nn.Module):
        if type(module) == torch.nn.Conv2d:
            log.debug(f'Initializing conv weights')
            torch.nn.init.normal_(module.weight, mean=0, std=.02)
        if type(module) == torch.nn.Linear:
            log.debug(f'Initializing FC weights')
            torch.nn.init.normal_(module.weight, mean=0, std=.02)
            torch.nn.init.normal_(module.bias, mean=0, std=.02)

    classifier = torchvision.models.resnet50()
    # The pre-built resnet model is configured to classify 1000 ImageNet classes - change the final layer to match our problem
    classifier.fc = torch.nn.Linear(in_features=2048, out_features=num_labels, bias=True)
    if args.pretrained_weights:
        log.debug(f'Loading classifier from {args.pretrained_weights}')
        classifier.load_state_dict(torch.load(args.pretrained_weights))
    else:
        classifier.apply(init_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f'Using device: {device}')
    classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = ignite.engine.create_supervised_trainer(classifier, optimizer, criterion, device=device)

    val_metrics = {
        "accuracy": ignite.metrics.Accuracy(),
        "f1": ignite.metrics.Fbeta(1),
        "cross_entropy": ignite.metrics.Loss(criterion),
        "precision": ignite.metrics.precision.Precision(),
        "recall": ignite.metrics.recall.Recall(),
    }
    evaluator = ignite.engine.create_supervised_evaluator(classifier, metrics=val_metrics, device=device)

    to_save = {
        'classifier': classifier,
    }
    checkpoint_handler = ignite.handlers.Checkpoint(
        to_save,
        save_handler=args.model_save_dir,  # Saves to this dir
        n_saved=5,
        filename_prefix='best',
        score_function=lambda engine: -engine.state.metrics['cross_entropy'],
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(ignite.engine.Events.COMPLETED, checkpoint_handler)

    handler = ignite.handlers.early_stopping.EarlyStopping(
        patience=args.early_stopping_patience,
        score_function=lambda engine: -engine.state.metrics['cross_entropy'],
        trainer=trainer,
    )
    evaluator.add_event_handler(ignite.engine.Events.COMPLETED, handler)

    tboard_summary_writer = SummaryWriter(log_dir=args.tboard_log_dir)

    def tboard_log_fn(metric, value, epoch, prefix):
        tboard_summary_writer.add_scalar(f"{prefix}/{metric}", value, epoch)

    def logger_log_fn(metric, value, prefix):
        log.info(f"{prefix}/{metric} = {value}")

    def log_metrics(metrics, log_fns=[]):
        for metric, value in metrics.items():
            try:
                for i, v in enumerate(value):
                    for log_fn in log_fns:
                        log_fn(f'{metric}/{i}', v)
            except TypeError:  # Not iterable, hopefully
                for log_fn in log_fns:
                    log_fn(metric, value)

    train_batch_size = val_batch_size = test_batch_size = 100
    log.info(f'Batch sizes: train_batch_size={train_batch_size} and val_batch_size={val_batch_size}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, pin_memory=True, shuffle=True,
                                               num_workers=args.num_dataloader_threads_per_dataset)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, pin_memory=True, shuffle=True,
                                             num_workers=args.num_dataloader_threads_per_dataset)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, pin_memory=True, shuffle=True,
                                              num_workers=args.num_dataloader_threads_per_dataset)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        tboard_log_fn('loss', engine.state.output, engine.state.epoch, 'train')
        evaluator.run(val_loader)

    @evaluator.on(ignite.engine.Events.COMPLETED)
    def log_validation_results(engine):
        def val_tboard_logger(metric, value):
            tboard_log_fn(metric, value, trainer.state.epoch, 'val')
        log_metrics(engine.state.metrics, log_fns=[val_tboard_logger])

    trainer.run(train_loader, max_epochs=args.num_epochs)

    log.info(f'Loading best classifier from {checkpoint_handler.last_checkpoint}')
    classifier.load_state_dict(torch.load(checkpoint_handler.last_checkpoint))
    test_evaluator = ignite.engine.create_supervised_evaluator(classifier, metrics=val_metrics, device=device)
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    tboard_logger = lambda metric, value: tboard_log_fn(metric, value, trainer.state.epoch, 'test')
    console_logger = lambda metric, value: logger_log_fn(metric, value, 'test')
    log_metrics(metrics, log_fns=[tboard_logger, console_logger])
