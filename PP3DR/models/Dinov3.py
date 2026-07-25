import torch
import torchvision.transforms.v2 as transforms
from transformers.image_utils import load_image
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np

image_to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])


def load_dinov3(model = "facebook/dinov3-vith16plus-pretrain-lvd1689m", token = "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"):
    """@Returns: processor, model"""
    # Don't specify the dtype. Allow the model to use AMP.
    return AutoImageProcessor.from_pretrained(model, token=token), AutoModel.from_pretrained(model, token=token)


def obtain_features(processor, model, image, remove_registers = True):
    """@Returns: (batch size, # patches, 1280)"""
    if remove_registers:
        with torch.no_grad():
            return model(**processor(image, return_tensors="pt", do_resize=False, do_center_crop=False)).last_hidden_state[:, 5:]
    else:
        with torch.no_grad():
            return model(**processor(image, return_tensors="pt", do_resize=False, do_center_crop=False)).last_hidden_state


def low_rank(features, dim = 3):
    """@Returns: (batch size, # patches, dim)"""
    U, S, V = torch.pca_lowrank(features.to(torch.float32), dim)
    return features @ V


def write_to_image(image, h, w, name = "image.png"):
    f_min = image.min()
    f_max = image.max()
    features = (image - f_min) / (f_max - f_min) * 255
    cv2.imwrite(name, features.reshape(h, w, 3).detach().cpu().numpy().astype(np.uint8))


if __name__ == "__main__":
    import time
    import torchvision

    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
    image = transforms.functional.to_dtype(torchvision.io.decode_image("/vulcanscratch/hughma/Pi3/data/sintel/training/final/alley_1/frame_0001.png"), torch.float32, scale=True)
    image = image.unsqueeze(0)
    processor, model = load_dinov3()
    processor, model = processor, model.to("cuda").eval()
    print(f"original shape: {image.size}")
    # image = image.crop((0, 0, image.size[0] // 16 * 16, image.size[1] // 16 * 16))
    print(f"resized shape: {image.size}")
    data = image_to_tensor(image).to("cuda", non_blocking = True)
    start = time.time()
    features = obtain_features(processor, model, data)[0]
    end = time.time()
    print("Elapsed time (slight overestimation):", end - start)
    print(features.shape)
    features = low_rank(features)
    print(f"lowrank: {features.shape}")
    # write_to_image(features, 42, 60)
    write_to_image(features, 27, 64, name = "Dino_sintel.png")
