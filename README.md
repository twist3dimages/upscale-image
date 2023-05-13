# Upscale-Image

This repository contains a Dockerfile that creates a Docker image to upscale images using AI models. The Python script uses an AI model trained for super-resolution tasks to upscale images. It can be run on a device with CUDA support to utilize the GPU for faster processing.

## Building the Docker Image

You can build the Docker image using the Docker CLI:

```shell
docker build -t upscale-image .
```shell

# Upscaling Photos
To upscale photos, follow these steps:

Create a directory on your host machine and place the photos you want to upscale in this directory.

Run the Docker image, mounting the directory with the input photos to the /app/input directory inside the container:

```shell
docker run -it --rm -v /path/to/input:/app/input -v /path/to/output:/app/output upscale-image
Replace /path/to/input with the path to the directory containing the input photos on your host machine. Similarly, replace /path/to/output with the path to the directory where you want the upscaled photos to be saved.

The script will process the photos and save the upscaled versions to the specified output directory.
Note: By default, the script assumes that the input photos are in JPEG format. If your photos have a different format, you may need to modify the script or install additional dependencies as required.

Remember to replace `/path/to/input` and `/path/to/output` with actual paths on yo
