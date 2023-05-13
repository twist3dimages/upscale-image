import cv2
import os
import torch
from model import RRDB_Net
import numpy as np
import argparse

def upscale_image(img_path, output_path, model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = RRDB_Net(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_path, output)

def upscale_images_in_dir(input_dir, output_dir, models_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_paths = [os.path.join(models_dir, model_name) for model_name in os.listdir(models_dir) if model_name.endswith('.pth')]

    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            print(f'Processing {img_name}')
            for model_path in model_paths:
                output_img_name = f'{os.path.splitext(img_name)[0]}_{os.path.basename(model_path)}.png'
                upscale_image(os.path.join(input_dir, img_name),
                              os.path.join(output_dir, output_img_name),
                              model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upscale images using RRDB_Net models')
    parser.add_argument('--input', type=str, help='Path to the input directory')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--models', type=str, help='Path to the models directory')
    args = parser.parse_args()

    input_dir = args.input or '/input'
    output_dir = args.output or '/output'
    models_dir = args.models or '/app/models'
    upscale_images_in_dir(input_dir, output_dir, models_dir)
