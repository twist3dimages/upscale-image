# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update the system and install OpenCV and other necessary libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install opencv-python numpy

# Make port 80 available to the world outside this container
EXPOSE 80

# Run upscale.py when the container launches
CMD ["python", "upscale.py"]
