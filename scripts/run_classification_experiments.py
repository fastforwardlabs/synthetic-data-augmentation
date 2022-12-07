"""
Creates a chain of dependent jobs on CDSW in order to run a set of classification experiments, with varying amounts of synthetic data.
"""
import json
import os
import cmlapi

# instantiate CML API client
client = cmlapi.default_client()

# get project id
projects = client.list_projects(search_filter=json.dumps({"name": "synthetic-data-augmentation"}))
PROJECT_ID = projects.projects[0].id

# get runtime id
py39_gpu_runtimes = client.list_runtimes(
    search_filter=json.dumps(
        {
            "kernel": "Python 3.9",
            "edition": "Nvidia GPU",
            "editor": "Workbench",
            "version": "2022.04",
            "image_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-cuda:2022.04.1-b6",
        }
    )
)
RUNTIME_ID = py39_gpu_runtimes.runtimes[0].image_identifier

try:
    module_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(module_base, 'scripts')
except NameError:
    # __file__ not defined
    # On CDSW be sure to set this environment variable to point to the dir containing the project scripts
    script_path = os.environ['SCRIPTS_PATH']
    module_base = os.path.dirname(script_path)

JOBS_SPECS = [
    # {
    #     'spec_name': 'Defect Proportion 0.1',
    #     'train_csv': os.path.join(module_base, 'data', 'train.0.1.csv'),
    #     'cyclegan_defective_csv': os.path.join(module_base, 'data', 'train_defective_only.0.1.csv'),
    #     'cyclegan_undefective_csv': os.path.join(module_base, 'data', 'train_undefective_only.csv'),
    #     'cyclegan_model_save_dir': 'models/cyclegan/class3/0.1/',
    #     'cyclegan_tboard_dir': 'logs/cyclegan/class3/0.1/',
    #     'synthetic_image_batches': (5308 - 532),
    #     'synthetic_image_dir': os.path.join(module_base, 'data', 'synth', '0.1'),
    #     'synthetic_csv': os.path.join(module_base, 'data', 'synth', 'train.0.1.csv'),
    #     'classifier_save_dir': 'models/classifier/class3/0.1/',
    #     'classifier_tboard_dir': 'logs/classifier/class3/0.1/',
    # },
    #
    # {
    #     'spec_name': 'Defect Proportion 0.5',
    #     'train_csv': os.path.join(module_base, 'data', 'train.0.5.csv'),
    #     'cyclegan_defective_csv': os.path.join(module_base, 'data', 'train_defective_only.0.5.csv'),
    #     'cyclegan_undefective_csv': os.path.join(module_base, 'data', 'train_undefective_only.csv'),
    #     'cyclegan_model_save_dir': 'models/cyclegan/class3/0.5/',
    #     'cyclegan_tboard_dir': 'logs/cyclegan/class3/0.5/',
    #     'synthetic_image_batches': (5308 - 2655),
    #     'synthetic_image_dir': os.path.join(module_base, 'data', 'synth', '0.5'),
    #     'synthetic_csv': os.path.join(module_base, 'data', 'synth', 'train.0.5.csv'),
    #     'classifier_save_dir': 'models/classifier/class3/0.5/',
    #     'classifier_tboard_dir': 'logs/classifier/class3/0.5/',
    # },
    #
    # {
    #     'spec_name': 'Defect Proportion 0.25',
    #     'train_csv': os.path.join(module_base, 'data', 'train.0.25.csv'),
    #     'cyclegan_defective_csv': os.path.join(module_base, 'data', 'train_defective_only.0.25.csv'),
    #     'cyclegan_undefective_csv': os.path.join(module_base, 'data', 'train_undefective_only.csv'),
    #     'cyclegan_model_save_dir': 'models/cyclegan/class3/0.25/',
    #     'cyclegan_tboard_dir': 'logs/cyclegan/class3/0.25/',
    #     'synthetic_image_batches': (5308 - 1328),
    #     'synthetic_image_dir': os.path.join(module_base, 'data', 'synth', '0.25'),
    #     'synthetic_csv': os.path.join(module_base, 'data', 'synth', 'train.0.25.csv'),
    #     'classifier_save_dir': 'models/classifier/class3/0.25/',
    #     'classifier_tboard_dir': 'logs/classifier/class3/0.25/',
    # },
    {
        'spec_name': 'Defect Proportion 0.75',
        'train_csv': os.path.join(module_base, 'data', 'train.0.75.csv'),
        'cyclegan_defective_csv': os.path.join(module_base, 'data', 'train_defective_only.0.75.csv'),
        'cyclegan_undefective_csv': os.path.join(module_base, 'data', 'train_undefective_only.csv'),
        'cyclegan_model_save_dir': 'models/cyclegan/class3/0.75/',
        'cyclegan_tboard_dir': 'logs/cyclegan/class3/0.75/',
        'synthetic_image_batches': (5308 - 3980),
        'synthetic_image_dir': os.path.join(module_base, 'data', 'synth', '0.75'),
        'synthetic_csv': os.path.join(module_base, 'data', 'synth', 'train.0.75.csv'),
        'classifier_save_dir': 'models/classifier/class3/0.75/',
        'classifier_tboard_dir': 'logs/classifier/class3/0.75/',
    },
    {
        'spec_name': 'Defect Proportion 0.9',
        'train_csv': os.path.join(module_base, 'data', 'train.0.9.csv'),
        'cyclegan_defective_csv': os.path.join(module_base, 'data', 'train_defective_only.0.9.csv'),
        'cyclegan_undefective_csv': os.path.join(module_base, 'data', 'train_undefective_only.csv'),
        'cyclegan_model_save_dir': 'models/cyclegan/class3/0.9/',
        'cyclegan_tboard_dir': 'logs/cyclegan/class3/0.9/',
        'synthetic_image_batches': (5308 - 4776),
        'synthetic_image_dir': os.path.join(module_base, 'data', 'synth', '0.9'),
        'synthetic_csv': os.path.join(module_base, 'data', 'synth', 'train.0.9.csv'),
        'classifier_save_dir': 'models/classifier/class3/0.9/',
        'classifier_tboard_dir': 'logs/classifier/class3/0.9/',
    },
]

last_job = None
for job_spec in JOBS_SPECS:
    # 1. Train CycleGAN
    train_cyclegan_args = "--y_windows /home/cdsw/synthetic-data-augmentation/data/train_undefective_only.csv "\
                          f"--x_windows {job_spec['cyclegan_defective_csv']} "\
                          "--early_stopping_patience 200 --disc_lr_factor 5 --num_batches 400 --read_grayscale "\
                          "--num_dataloader_threads_per_dataset 1 --num_epochs 200 --x_class 3 "\
                          f"--model_save_dir {job_spec['cyclegan_model_save_dir']} "\
                          f"--tboard_log_dir {job_spec['cyclegan_tboard_dir']}"
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Train CycleGAN - {job_spec['spec_name']}",
        script=os.path.join('synthetic-data-augmentation', 'scripts', 'train_cyclegan.py'),
        runtime_identifier=RUNTIME_ID,
        cpu=2.0,
        memory=4.0,
        nvidia_gpu=1,
        arguments=train_cyclegan_args,
        parent_job_id=None if not last_job else last_job.id,
    )
    last_job = client.create_job(job_body, PROJECT_ID)

    # 2. Create synthetic images
    create_synth_images_args = f"--pretrained_weights {job_spec['cyclegan_model_save_dir']}/y_gen.199.pth "\
                               f"--batch_size 1 --head {job_spec['synthetic_image_batches']} --read_grayscale "\
                               f"--output_dir {job_spec['synthetic_image_dir']} "\
                               f"--output_csv {job_spec['synthetic_csv']}"
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Create Synthetic Images - {job_spec['spec_name']}",
        script=os.path.join('synthetic-data-augmentation', 'scripts', 'create_fake_defective_images.py'),
        runtime_identifier=RUNTIME_ID,
        cpu=2.0,
        memory=4.0,
        nvidia_gpu=1,
        arguments=create_synth_images_args,
        parent_job_id=None if not last_job else last_job.id,
    )
    last_job = client.create_job(job_body, PROJECT_ID)

    # 3. Train classifiers
    baseline_classifier_args = f"--train_csv {job_spec['train_csv']} "\
                               f"--early_stopping_patience 30 --lr 1e-5 --num_epochs 100 "\
                               f"--model_save_dir {job_spec['classifier_save_dir']}/baseline "\
                               f"--tboard_log_dir {job_spec['classifier_tboard_dir']}/baseline "
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Train Baseline Classifier - {job_spec['spec_name']}",
        script=os.path.join('synthetic-data-augmentation', 'scripts', 'train_classifier.py'),
        runtime_identifier=RUNTIME_ID,
        cpu=2.0,
        memory=4.0,
        nvidia_gpu=1,
        arguments=baseline_classifier_args,
        parent_job_id=None if not last_job else last_job.id,
    )
    last_job = client.create_job(job_body, PROJECT_ID)

    oversampled_classifier_args = "--oversample_minority_class " \
                                  f"--train_csv {job_spec['train_csv']} " \
                                  f"--early_stopping_patience 30 --lr 1e-5 --num_epochs 100 "\
                                  f"--model_save_dir {job_spec['classifier_save_dir']}/oversampled "\
                                  f"--tboard_log_dir {job_spec['classifier_tboard_dir']}/oversampled "
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Train Oversampled Classifier - {job_spec['spec_name']}",
        script=os.path.join('synthetic-data-augmentation', 'scripts', 'train_classifier.py'),
        runtime_identifier=RUNTIME_ID,
        cpu=2.0,
        memory=4.0,
        nvidia_gpu=1,
        arguments=oversampled_classifier_args,
        parent_job_id=None if not last_job else last_job.id,
    )
    last_job = client.create_job(job_body, PROJECT_ID)

    synthetic_classifier_args = f"--synthetic_image_base {job_spec['synthetic_image_dir']} " \
                                f"--synthetic_csv {job_spec['synthetic_csv']} " \
                                f"--train_csv {job_spec['train_csv']} " \
                                f"--early_stopping_patience 30 --lr 1e-5 --num_epochs 100 "\
                                f"--model_save_dir {job_spec['classifier_save_dir']}/synthetic "\
                                f"--tboard_log_dir {job_spec['classifier_tboard_dir']}/synthetic "
    job_body = cmlapi.CreateJobRequest(
        project_id=PROJECT_ID,
        name=f"Train Synthetic Classifier - {job_spec['spec_name']}",
        script=os.path.join('synthetic-data-augmentation', 'scripts', 'train_classifier.py'),
        runtime_identifier=RUNTIME_ID,
        cpu=2.0,
        memory=4.0,
        nvidia_gpu=1,
        arguments=synthetic_classifier_args,
        parent_job_id=None if not last_job else last_job.id,
    )
    last_job = client.create_job(job_body, PROJECT_ID)
