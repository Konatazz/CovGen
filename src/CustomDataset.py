from PIL import Image
from torch.utils.data import Dataset
import torch
from helpers.image_rescale import reduce_image
import pickle
import os
import numpy as np
import math


class CustomDataset(Dataset):
    """Generative Dataset."""

    def __init__(self, data_path, num_data, cls_min, selected_classes=None, transform=True, shuffle=True, scale=None, loadMem=False):
        """
        Args:
            data_path (str): Path to the data to load in
            num_data (int): Total number of data points to train the model on
            cls_min (int): The min class value in the data
            selected_classes (list or None): List of selected class indices to use
            transform (boolean): Transform data between -1 and 1
            shuffle (boolean): True to shuffle the data upon entering. False otherwise
            scale (str or NoneType): Scale data "up" or "down" to the nearest power of 2
                                     or keep the data the same shape with None
            loadMem (boolean): True to load in all data to memory, False to keep it on disk
        """

        # Save the data information
        self.data_path = data_path
        self.num_data = num_data
        self.transform = transform
        self.scale = scale
        self.loadMem = loadMem

        self.selected_classes = selected_classes

        # The min class value represents the value that needs to be
        # subtracted from the class value so the min value will be 0
        self.cls_scale = cls_min

        # Load in all the data onto the disk if specified
        if self.loadMem:
            # Load in the massive data tensors
            self.data_mat = torch.load("data/Imagenet64_imgs.pt")
            self.label_mat = torch.load("data/Imagenet64_labels.pt")

            # Get the number of data loaded in
            self.num_data = self.data_mat.shape[0]

            # Make sure the labels and data have the same shapes
            assert self.data_mat.shape[0] == self.label_mat.shape[0]

            print(f"{self.num_data} data loaded in")

            if self.selected_classes is not None:
                mask = np.isin(self.label_mat, self.selected_classes)
                self.data_mat = self.data_mat[mask]
                self.label_mat = self.label_mat[mask]
                self.num_data = self.data_mat.shape[0]
                print(f"Filtered to {self.num_data} data points for selected classes.")

                unique_classes = np.unique(self.label_mat)
                print(f"Actual imported classes: {unique_classes}")

        # Create a list of indices which can be used to
        # essentially shuffle the data
        self.data_idxs = np.arange(0, self.num_data)
        if shuffle:
            np.random.shuffle(self.data_idxs)

    def get_first_n_classes_images(self, n_classes=5, n_images_per_class=10, output_dir='output_images'):

        os.makedirs(output_dir, exist_ok=True)
        unique_classes = np.unique(self.label_mat)
        selected_classes = unique_classes[:n_classes]
        images_per_class = {cls: [] for cls in selected_classes}

        for idx in range(len(self)):
            image, label = self[idx]

            if label.item() in selected_classes:
                images_per_class[label.item()].append(image)

                if len(images_per_class[label.item()]) >= n_images_per_class:
                    continue

        for cls, images in images_per_class.items():
            for i, img in enumerate(images):
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np + 1) / 2 * 255
                img_np = img_np.astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.save(os.path.join(output_dir, f'class_{cls}_img_{i}.png'))
        print("Images saved successfully***************************************************************************************************.")
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Convert the given index to the shuffled index
        data_idx = self.data_idxs[idx]

        # If the files were pre-loaded into memory,
        # just grab them from meory
        if self.loadMem == True:
            image = self.data_mat[data_idx].clone()
            label = self.label_mat[data_idx].clone()

            # Subtract the min class value so the min label is 0
            label -= self.cls_scale
        
        # If the files are not preloaded, then
        # get them from disk individually
        else:
            # Open the data file and load it in
            data = pickle.load(open(f"{self.data_path}{os.sep}{data_idx}.pkl", "rb"))

            # Get the image and class label from the data
            image = data["img"]
            label = data["label"]

            # Subtract the min class value so the min label is 0
            label -= self.cls_scale

            # Convert the data to a tensor
            image = torch.tensor(image, dtype=torch.float32, device=torch.device("cpu"))
            image = image.reshape(3, 64, 64)
            label = torch.tensor(label, dtype=torch.int)

        # Reshape the image to the nearest power of 2
        if self.scale is not None:
            if self.scale == "down":
                next_power_of_2 = 2**math.floor(math.log2(image.shape[-1]))
            elif self.scale == "up":
                next_power_of_2 = 2**math.ceil(math.log2(image.shape[-1]))
            image = torch.nn.functional.interpolate(image, (next_power_of_2, next_power_of_2))

        # Transform the image between -1 and 1
        if self.transform:
            image = reduce_image(image)

        # Return the image and label
        return image,label