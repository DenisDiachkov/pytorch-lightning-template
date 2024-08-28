# PyTorch Lightning Template

This repository provides a PyTorch Lightning-based training pipeline.

## Features

- PyTorch Lightning integration.
- WanDB integration for logging of metrics, code, and visualisation.
- CI/CD via GitHub Actions.
  - Automated tests with examples.
  - Automated Docker image build and push.
  - Automated PyPI package build and push.
  - Automated versioning.

## Project Structure

- `src/pytorch_lightning_template`: Python package.
  - `__init__.py`: Package initialization.
  - `__main__.py`: Main module logic.
  - `py.typed`: Marker file for PEP 561 typing.
  - `module.py`: LightningModule for training and evaluation of torch models.
  - `model/`: Model module, where torch models are defined.
    - `__init__.py`
    - ...
  - `dataset/`: Dataset module, where data classes are defined. 
    - `__init__.py`: Dataset initialization.
    - `datamodule.py`: Generic datamodule for PyTorch Lightning.
    - ...
  - `criterion/`: Criterion module, where loss functions are defined.
    - `__init__.py`
    - `multi_criterion.py`: Generic multi-loss criterion.
  - `utils/`: Utility module.
  - `train.py`: Training script. Called by `__main__.py`.
  - `test.py`: Testing script. Called by `__main__.py`.
- `pyproject.toml`: Specifies build system requirements and project dependencies.
- `tests/`: Unit and functional tests.
  - `unittests/`: Unit tests. Run with `pytest tests/unittests/`.
  - `functional_tests/`: Functional tests. Run with `pytest tests/functional_tests/`.
  - `test_data/`: Test data.
  - `conftest.py`: Pytest configuration.
- `.github/workflows/`: 
  - `build-analysis-test.yml`: CI for build and test.
  - `build-pypi-docker.yml`: CI for Docker and PyPI deployment.
- `dockerfile`: Docker setup for Python.
- `.isort.cfg`: isort configuration.
- `.pylintrc`: Pylint configuration.
- `LICENSE.txt`: License file.
- `VERSION`: Version file. Updated by CI/CD. 

## Getting Started

### CICD Setup

For testing and deployment in a CI/CD setup, refer to our [CICD project](https://github.com/AGISwarm/CICD).
Pip package and docker images are pushed to a self-hosted PyPI server and Docker registry.
Stick to git installation if you don't need CI/CD.


### Installation

- Installation from git

   1. Clone the repository:
      ```bash
      git clone https://github.com/DenisDiachkov/pytorch-lightning-template.git
      ```
   2. Navigate to the project directory:
      ```bash
      cd pytorch-lightning-template
      ```
   3. Install dependencies:
      ```bash
      pip install -e .  # Install package in editable mode is recommended for development.
      ```

- Installation from PyPI

  1. Install the package:
     ```bash
     pip install pytorch-lightning-template --extra-index-url http://pypi-server/
     ```

- Installation from Docker

  1. Pull the Docker image:
     ```bash
     docker pull docker-registry:80/pytorch_lightning_template:0.1.0
     ```
   
### WanDB Setup

1. Create a free account on [WanDB](https://wandb.ai/).
2. Install WanDB (should be installed with the package):
   ```bash
   pip install wandb
   ```
3. Login to WanDB:
   ```bash
   wandb login
   ```

### Usage

1. Run the package with a specified configuration file for training or testing:
   ```bash
   python -m pytorch_lightning_template --cfg <path to config file>
   ```

### Training Configuration

```yaml
mode: !!str "train"  # train or test
wall: !!bool False  # If True, all warnings will be treated as errors 
seed: !!int &seed 42  # Random seed
experiment_name: &experiment_name <your experiment name> # Experiment name for WanDB logging and checkpoint saving (will be saved to ./experiments/*experiment_name*)
version: &version 0
resume_path:  # Checkpoint .ckpt path to resume the training
no_logging: False  # If True, turns off the WanDB logging
loglevel: !!str "debug"  # Loglevel for python logging (debug, info, warning, error, critical)

environ_vars:  # System environment variables 
  WANDB_SILENT: !!bool False

logger_params:  # WanDB logger parameters
  project: !!str "<your project name>"
  name: *experiment_name
  version: null
  save_dir: "/tmp/wandb/"
 
trainer_params:  # PyTorch Lightning trainer parameters. See more here: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
  deterministic: !!str "warn"
  devices: [0, 1]
  accelerator: !!str "cuda"
  num_sanity_val_steps: !!int 2
  max_epochs: 100
  precision: 32
  limit_train_batches: null
  limit_val_batches: !!int 10
  log_every_n_steps: !!int &log_metrics_every_n_steps 5

trainer_callbacks:  # PyTorch Lightning callbacks. See more here: https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html
  [
    {
      callback: pytorch_lightning.callbacks.early_stopping.EarlyStopping, # Callback class
      callback_params: # Callback parameters
        {
          monitor: !!str "val_loss",
          min_delta: !!float 0.0001,
          patience: !!int 100000,
          verbose: !!bool False,
          mode: !!str "min",
        },
    },
    {
      callback: pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
      callback_params:
        {
          monitor: !!str "val_loss",
          filename: !!str "best_Epoch={epoch}_Loss={val_loss:.2f}",
          save_top_k: !!int 1,
          save_last: !!bool True,
          mode: !!str "min",
          verbose: !!bool False,
        },
    },
  ]

lightning_module: .module.BaseModule  # PyTorch Lightning module class
lightning_module_params:  # PyTorch Lightning module parameters
  model: .model.resnet.ResNet  # Torch model class
  model_params:  # Torch model parameters
    in_channels: !!int 1
    out_channels: !!int 64
    num_blocks: !!int 3
    num_classes: !!int 10
    

  optimizer: torch.optim.AdamW  # Torch optimizer class
  optimizer_params:  # Torch optimizer parameters
    lr: !!float 0.001
  criterion: .criterion.multi_criterion.MultiCriterion  # Torch criterion class
  criterion_params:  # Torch criterion parameters
    {
      criterions:
        [
          {
            criterion: torch.nn.CrossEntropyLoss,
            criterion_params: {}
          },
        ],
    }
  scheduler: torch.optim.lr_scheduler.CosineAnnealingLR  # Torch scheduler class
  scheduler_params:  # Torch scheduler parameters
    T_max: !!int 100
    eta_min: !!float 0.0001
  log_metrics_every_n_steps: *log_metrics_every_n_steps
  visualization_every_n_steps: !!int 20

datamodule_params:  # PyTorch Lightning datamodule parameters (Implementation here: src/pytorch_lightning_template/dataset/datamodule.py)
  dataset: .dataset.fashion_mnist.FashionMNIST
  dataset_params: {
    root: !!str './data',
  }
  train_dataset_params:  # Parameters specific to the training dataset (will override dataset_params)
    albumentations_transform:  # Albumentation A.Compose components (see more here: https://albumentations.ai/docs/api_reference/core/composition/)
      {
        albumentations.CropNonEmptyMaskIfExists:  # Augmentation class
          { height: 512, width: 512, p: 1.0 }, # Augmentation parameters
        albumentations.HorizontalFlip: { p: 0.5 },
        albumentations.ShiftScaleRotate: { p: 0.2 },
      }
  dataloader_params:  # DataLoader parameters
    shuffle: !!bool True
    num_workers: !!int 8
    pin_memory: !!bool True
    persistent_workers: !!bool True
  train_dataloader_params:  # Parameters specific to the training dataloader (will override dataloader_params)
    batch_size: !!int 8
    shuffle: !!bool True
  val_dataloader_params:   # Parameters specific to the validation dataloader (will override dataloader_params)
    batch_size: !!int 8
    shuffle: !!bool False

```

### Testing Configuration

See [here](cfg/test.yaml)

#### 

### Docker Building (if not using CI/CD)

1. Build python package:
   ```bash
   python -m build
   ```
2. (Optional) Create PIP_INDEX_EXTRA_URL environment variable to be able to install dependencies from a self-hosted PyPI server.

   Given URL is self-hosted PyPI server from [CICD](README.md#cicd-setup) setup. 
   Replace with your own if needed. 
   Or remove secret mounting in Dockerfile, if no need. 
   ```bash
   export PIP_INDEX_EXTRA_URL=http://pypi-server/
   ```
3. Build Docker image. Use this command for testing purposes only. If you use [CICD](README.md#cicd-setup) setup, it will build and push the image for you after creating a tag in your repository:
   ```bash
   docker build -t <your tag name> --secret id=PIP_INDEX_EXTRA_URL,env=PIP_INDEX_EXTRA_URL .
   ```
   or simply
   ```bash
   docker build -t <your tag name> .
   ```
   if you don't need a self-hosted PyPI server.

### Docker Run

1. [Pull the Docker image](README.md#installation-from-docker) or [build it](README.md#docker-building-if-not-using-cicd).
2. Run the Docker container:
   ```bash
   docker run pytorch_lightning_template:0.1.0 -v <path to config file>:/cfg/config.yaml
   ```


## Future Work

1. Separate inference ready model checkpoint from training checkpoints.
2. Develop a separate inference pipeline with a REST API, load-balancing, and advanced logging.
3. Implement Optuna wrapper for hyperparameter tuning.
4. Make WanDB logging optional, replaceable with other loggers.
5. Implement a more advanced configuration system with a configuration file schema. Prefferably strictly typed.
6. Remote model repository for model versioning and sharing.

## License

Distributed under the MIT License. See `LICENSE.txt` for details.

## Contact

Denis Diachkov - diachkov.da@gmail.com