import sys, os
sys.path.insert(0, os.path.abspath('src/'))
import torch
from torch import nn
from torchvision import transforms
import math
import numpy as np
from src.models.diff_model import diff_model

cpu = torch.device("cpu")

def compute_model_stats(
        model_dirname="",
        model_filename = "",
        model_params_filename = "",

        device = "gpu",
        gpu_num = 0,

        num_fake_imgs = 10000,
        batchSize = 20,

        step_size = 10,
        DDIM_scale = 1,
        corrected = True,

        file_path = "",
        mean_filename = "",
        var_filename = "",
    ):

    def normalize(imgs):
        imgs = transforms.Compose([transforms.Resize((299,299))])(imgs)
        imgs = imgs/255.0
        return transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(imgs)

    if device == "gpu":
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device(f"cpu")

    model = diff_model(3, 3, 1, 1, ["res", "res"], 100000, "cosine", 100, device, 100, 1000, 16, 0.0, step_size, DDIM_scale)
    model.loadModel(model_dirname, model_filename, model_params_filename)
    model.eval()

    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.DEFAULT")
    inceptionV3.eval()
    inceptionV3.to(model.device)

    inceptionV3.fc = nn.Identity()
    inceptionV3.aux_logits = False

    scores = None
    with torch.no_grad():
        for i in range(math.ceil(num_fake_imgs/batchSize)):
            cur_batch_size = min(num_fake_imgs, batchSize*(i+1))-batchSize*i
            imgs = model.sample_imgs(cur_batch_size, use_tqdm=True, unreduce=True, corrected=corrected)
            imgs = normalize(imgs.to(torch.uint8))
            if type(scores) == type(None):
                scores = inceptionV3(imgs).to(cpu).numpy().astype(np.float16)
            else:
                scores = np.concatenate((scores, inceptionV3(imgs).to(cpu).numpy().astype(np.float16)), axis=0)

            print(f"Num loaded: {min(num_fake_imgs, batchSize*(i+1))}")

    device = model.device
    del model, inceptionV3

    mean = np.mean(scores, axis=0)
    var = np.cov(scores, rowvar=False)

    np.save(f"{file_path}{os.sep}{mean_filename}", mean)
    np.save(f"{file_path}{os.sep}{var_filename}", var)

if __name__ == "__main__":
    compute_model_stats()