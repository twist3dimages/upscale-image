# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install apt-utils to address package configuration warning
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# Update the system and install OpenCV and other necessary libraries
RUN apt-get install -y libgl1-mesa-glx wget

# Create and activate the virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip within the virtual environment
RUN bash -c "source /venv/bin/activate && pip install --no-cache-dir --upgrade pip"

# Install additional pip modules inside the virtual environment
RUN bash -c "source /venv/bin/activate && pip install --no-cache-dir opencv-python numpy"

# Download RRDB_Net model
RUN wget -O /app/models/RRDB_ESRGAN_x4.pth https://github.com/xinntao/ESRGAN/releases/download/v0.1.1/RRDB_ESRGAN_x4.pth

# Download ESRGAN model
RUN wget -O /app/models/ESRGAN_SRx4_DF2K_official-ff704c30.pth https://github.com/xinntao/ESRGAN/releases/download/v0.4.0/ESRGAN_SRx4_DF2K_official-ff704c30.pth

# Make port 80 available to the world outside this container
EXPOSE 80

# Run upscale.py when the container launches
CMD ["python", "upscale.py"]
