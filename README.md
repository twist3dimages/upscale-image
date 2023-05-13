# Upscale-Image

This repository contains a Dockerfile that creates a Docker image to upscale images using AI models. The Python script uses an AI model trained for super-resolution tasks to upscale images. It can be run on a device with CUDA support to utilize the GPU for faster processing.

## Building the Docker Image

You can build the Docker image using the Docker CLI:

```shell
docker build -t upscale-image .
