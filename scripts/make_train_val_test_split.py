"""
Creates a stratified train-validation-test split of available data.
"""
import argparse
import logging
import os
import numpy as np
import pandas as pd


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()


if __name__ == "__main__":
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
        '--defective_windows',
        type=str,
        default=os.path.join(module_base, 'data', 'windows.csv'),
        help='File to specify subregions of images to use for the defective images.'
    )
    parser.add_argument(
        '--undefective_windows',
        type=str,
        default=os.path.join(module_base, 'data', 'undefective_windows.csv'),
        help='File to specify subregions of images to use for the undefective images.'
    )
    parser.add_argument(
        '--defect_class',
        type=int,
        action='append',
        default=None,
        help='Specify defect classes to use. Leave blank to use all.'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Proportion of the dataset to use in the training split. '
             'The remainder will be equally divided in val and test splits.'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Pandas random state seed for determinstic outcomes.'
    )
    parser.add_argument(
        '--dump_dir',
        type=str,
        default=os.path.join(module_base, 'data'),
        help='Directory where output csv files will be written.'
    )

    args = parser.parse_args()
    log.info(f'Received args: {args}')

    assert 0.0 <= args.train_split <= 1.0, "Invalid train split value (must be between 0 and 1)."

    defective_df = pd.read_csv(args.defective_windows, index_col=0)
    undefective_df = pd.read_csv(args.undefective_windows, index_col=0)

    defect_classes = defective_df.ClassId.unique()
    log.info(f'Defect classes: {defect_classes}')

    defect_class_counts = {
        defect_class: np.sum(defective_df.ClassId == defect_class)
        for defect_class in defect_classes
    }
    log.info(f'Defect class counts: {defect_class_counts}')

    if args.defect_class:
        log.info(f'Keeping only defect classes: {args.defect_class}')
        mask = defective_df.ClassId.isin(args.defect_class)
        defective_df = defective_df[mask]

    undefective_df['ClassId'] = -1
    total_df = pd.concat([defective_df, undefective_df], ignore_index=True)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    rs = args.random_state
    for label, group_df in total_df.sample(frac=1, random_state=rs).groupby(by='ClassId'):
        train_split = group_df.sample(frac=args.train_split, random_state=rs)
        remainder_df = group_df.loc[group_df.index.difference(train_split.index)]
        val_test_split_idx = remainder_df.shape[0] // 2
        train_df = pd.concat([train_df, train_split], ignore_index=True)
        val_df = pd.concat([val_df, remainder_df.iloc[:val_test_split_idx]], ignore_index=True)
        test_df = pd.concat([test_df, remainder_df.iloc[val_test_split_idx:]], ignore_index=True)

    train_df.to_csv(os.path.join(args.dump_dir, 'train.csv'))
    val_df.to_csv(os.path.join(args.dump_dir, 'val.csv'))
    test_df.to_csv(os.path.join(args.dump_dir, 'test.csv'))

    # This split is useful for training CycleGAN
    undefective_mask = train_df.ClassId == -1
    train_df[undefective_mask].to_csv(os.path.join(args.dump_dir, 'train_undefective_only.csv'))
    train_df[~undefective_mask].to_csv(os.path.join(args.dump_dir, 'train_defective_only.csv'))