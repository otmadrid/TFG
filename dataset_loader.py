from image_datasets import SyntheticUIBSelection
import matplotlib.pyplot as plt
width_crop = 250
height_crop = 250

# Create a dataset
dataset = SyntheticUIBSelection('C:/Users/otmad/OneDrive/Escritorio/TFG/UIB_SINGLE_IMAGE', # Path to the dataset
                                split='val', # Choose between 'train' and 'val'
                                name_noisy='_noisy', # Name of the noisy images folder. Should be val + name_noisy or train + name_noisy
                                dims_crop = [width_crop, height_crop])

# Get the first sample
image, target = dataset[0]

# Display the image and target side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(image)
axs[0].set_title('Random Crop')
axs[0].axis('off')

# Target image
axs[1].imshow(target)
axs[1].set_title('Target Image')
axs[1].axis('off')

plt.show()