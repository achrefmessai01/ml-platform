
# ML Stack

This repository provides Docker Compose configurations for deploying the following services:

- **Label Studio**: Data labeling tool
- **MinIO**: S3-compatible object storage
- **MLflow**: Experiment tracking and model registry
- **PostgreSQL**: Backend database for MLflow

All services are connected for seamless data labeling, storage, and experiment tracking.

## Workspace Structure

```
ml-stack/
├── docker-compose.yml
├── README.md
├── .env.example
├── mlflow/
│   └── Dockerfile
```

## Prerequisites

Before running the stack, ensure you have:

- [Docker](https://docs.docker.com/get-docker/) installed
- Linux or macOS (Windows is not officially supported)

## Setup Instructions

1. Copy `.env.example` to `.env` and update environment variables as needed.
2. Create ML_STACK_FOLDER at location specified in `.env` (default: `~/SSD_drive_1.8T/ml-stack/`).
3. Set permission for label studio folder:
   ```sh
   sudo chown -R 1001:1001 ${ML_STACK_FOLDER}/label_studio
   ```
4. Run `docker compose up` to start all services.

## Notes

- The following directories are required for data persistence:
	- `${ML_STACK_FOLDER}/label_studio` (Label Studio data)
	- `${ML_STACK_FOLDER}/minio/data` (MinIO data)
	- `${ML_STACK_FOLDER}/db_data` (PostgreSQL data)

If these folders do not exist, Docker will attempt to create them automatically. However, you may create them manually to ensure correct permissions and structure:

```sh
mkdir -p ~/SSD_drive_1.8T/ml-stack/label_studio
mkdir -p ~/SSD_drive_1.8T/ml-stack/minio/data
mkdir -p ~/SSD_drive_1.8T/ml-stack/db_data
```