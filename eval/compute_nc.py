import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import Inception_V3_Weights
from PIL import Image

model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Module) and not isinstance(layer, models.inception.InceptionAux):
        layer.register_forward_hook(get_activation(name))

def compute_neuron_coverage(image_folder, activation_threshold=0.0):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    coverage_results = {}
    total_neurons_count = 0

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            activations.clear()

            _ = model(img_tensor)

            for name, layer_activation in activations.items():
                activated_neurons = (layer_activation > activation_threshold).nonzero(as_tuple=True)[1]
                if name not in coverage_results:
                    coverage_results[name] = set()
                coverage_results[name].update(activated_neurons.tolist())
                current_layer_activated_count = len(activated_neurons)

                layer = dict(model.named_modules())[name]
                if isinstance(layer, torch.nn.Conv2d):
                    total_neurons_count += layer.out_channels
                elif isinstance(layer, torch.nn.Linear):
                    total_neurons_count += layer.out_features

    for name, activated_neurons in coverage_results.items():
        if name in total_neurons:
            coverage[name] = len(activated_neurons) / total_neurons[name]

    overall_covered_neurons_count = sum(len(coverage_results[name]) for name in coverage_results)
    overall_coverage = overall_covered_neurons_count / total_neurons_count if total_neurons_count > 0 else 0
    return coverage, overall_coverage, overall_covered_neurons_count, total_neurons_count, coverage_results

image_folder = ''

activation_threshold = 0.6

coverage, overall_coverage, covered_neurons_count, total_neurons_count, coverage_results = compute_neuron_coverage(image_folder, activation_threshold)

for layer_name, coverage_value in coverage.items():
    covered_neurons = len(coverage_results[layer_name])
    print(f"Layer: {layer_name}, Covered Neurons: {covered_neurons}, Neuron Coverage: {coverage_value:.4f}")

print(f"Overall Covered Neurons: {covered_neurons_count}, Total Neurons: {total_neurons_count}, Overall Neuron Coverage: {overall_coverage:.4f}")