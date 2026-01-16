# ml-platform

This is the central monorepo for all machine learning related projects and components.
For usage and instructions please refer to the readme in the individual project directories.

## Data storage

Data is stored in Hetzner s3 buckets which can be fused to the local filesystem for easy access. In order for this to work the password file has to be created like below.
The access key and secret can be created in the hetzner console.

```bash
echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ${HOME}/.passwd-s3fs
chmod 600 ${HOME}/.passwd-s3fs
```

Below you can see an example for fusing the plant-datasets bucket. 

```bash
mkdir plant-datasets
s3fs plant-datasets ./plant-datasets -o passwd_file=${HOME}/.passwd-s3fs -o url=hel1.your-objectstorage.com -o use_path_request_style
```

https://github.com/s3fs-fuse/s3fs-fuse

For unmounting the bucket use:

```bash
fusermount -u plant-datasets
```

## Experiments

Experiments are tracked using MLflow. We are using the remote mlflow instance integrated into Gitlab. It can be accesed via this URI: `https://gitlab.com/api/v4/projects/72607084/ml/mlflow/`. You have to also provide an access token for authentication with at least the `api` scope. See: `https://docs.gitlab.com/user/project/ml/experiment_tracking/mlflow_client/#enable-mlflow-client-integration`.

## Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality and consistency. The hooks are configured to run automatically before each commit.

### Setup

Install pre-commit hooks:

```bash
pre-commit install
```

### Usage

The hooks will run automatically on `git commit`. To manually run all hooks on all files:

```bash
pre-commit run --all-files
```

The configured hooks include:
- **Ruff**: Python linter and formatter
- **Trailing whitespace**: Removes trailing whitespace
- **End of file fixer**: Ensures files end with a newline
- **YAML checker**: Validates YAML files
- **Large files checker**: Prevents accidentally committing large files
- **Merge conflict checker**: Detects merge conflict markers
- **Private key detector**: Prevents committing private keys
- **Submodules forbidder**: Prevents adding git submodules

## Workflow

The general workflow for developing and training models is as follows:

1. Create a new branch for your work.
2. Implement your model and training code in the appropriate project directory.
3. Train your model on whatever compute resources you have available.
4. Log your experiments using MLflow.
5. Log the model to the MLflow model registry.
6. Once you have an experiment run you are satisfied with, promote the run to a registered model version in the model registry.
7. Merge your branch back into the main branch.
