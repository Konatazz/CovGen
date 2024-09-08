import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def ssim(img1, img2, C1=1e-10, C2=1e-10):
    img1 = img1.mean(dim=1, keepdim=True)
    img2 = img2.mean(dim=1, keepdim=True)
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return numerator / denominator

folder1 = ''
folder2 = ''

images1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
images2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])

if len(images1) != len(images2):
    raise ValueError("")

ssim_values = []
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

for i in range(len(images1)):
    img1_path = os.path.join(folder1, images1[i])
    img2_path = os.path.join(folder2, images2[i])
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    ssim_value = ssim(img1_tensor, img2_tensor).item()
    ssim_values.append(ssim_value)

average_ssim = sum(ssim_values) / len(ssim_values)
