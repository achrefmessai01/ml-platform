# ml-platform

This is the central monorepo for all machine learning related projects and components.
For usage and instructions please refer to the readme in the individual project directories.

## Datasets

Datasets are stored on minio which is running on the vGreens server.
If you want to use a dataset for training, you can fuse the datasets bucket to the local filesystem using the following command:

```bash
mkdir datasets
s3fs datasets $(pwd)/datasets -o passwd_file=${HOME}/.passwd-s3fs -o url=http://localhost:9000/ -o use_path_request_style
```

Replace localhost with the server IP if you are accessing it remotely.
You also have to make sure to have an appropriate password file in the right place.
For more information on fusing please refer to:
https://github.com/s3fs-fuse/s3fs-fuse

For unmounting the datasets bucket use:

```bash
fusermount -u datasets
```

## Experiments

Experiments 
