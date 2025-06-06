import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, gridspec, pyplot as plt
import numpy as np


import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, gridspec, pyplot as plt
import numpy as np
import netCDF4 as nc  # Assuming 'nc' is for h5py (NetCDF is often handled via h5py in Python)


def plot_img(img):
    """
    Helper function to plot a single image.
    """
    #norm = colors.Normalize(-1, 1)
    plt.imshow(img, cmap='viridis')
    plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def plot_samples(gen, batch_gen, noise_gen, file_path, num_labels=10, samples_per_label=5, out_fn=None):
    ds = nc.Dataset(file_path)  # Replace with your actual NetCDF file path
    neval = 1100
    neval_idx = ds.variables['patch_start_indices'][neval]
    sla = ds.variables['sla'][neval_idx:neval_idx + 10, :, :]
    sla_lat = sla.shape[1]
    sla_lon = sla.shape[2]
    ds.close()

    conditioning_images = sla.reshape(10, sla_lat, sla_lon, 1)
    conditioning_images_repeated = np.repeat(conditioning_images, samples_per_label, axis=0) # shape (num_labels * samples_per_label, 16, 16, 1)
    
    # Create a set of labels for the samples, repeated samples_per_label times
    # labels = np.concatenate([np.array([l] * samples_per_label) for l in range(num_labels)])

    cond = conditioning_images_repeated
    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = num_labels * samples_per_label
        noise = next(noise_gen)[0]  # Noise batch of size (num_labels * samples_per_label, noise_dim)
    finally:
        noise_gen.batch_size = old_batch_size
    #print("Shapes:", cond.shape,noise.shape)
    # Generate images from the generator: inputs are (conditioning_images, noise)
    generated_images = gen.predict([cond, noise])
    #print('shapes',generated_images.shape, cond.shape, noise.shape)
    # Plot the samples in a grid
    plt.figure(figsize=(1.5 * num_labels, 1.5 * samples_per_label))

    # Create a grid for plotting
    gs = gridspec.GridSpec(samples_per_label+1, num_labels, hspace=0.02, wspace=0.02)

    for k in range(samples_per_label * num_labels):
        j = k // samples_per_label
        i = k % samples_per_label

        # Plot generated image
        plt.subplot(gs[i+1, j])
        plot_img(generated_images[k, :, :, 0])  # Plot the generated image

        # Optionally plot the conditioning image as well (in a smaller size)
    for k in range(num_labels):
        plt.subplot(gs[0, k])
        #print(k*samples_per_label)
        plot_img(cond[k*samples_per_label, :, :, 0])  # Conditioning image


    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()

def predict_image(gen, noise_gen, conditioning_image):
    conditioning_image = conditioning_image.reshape(1, *conditioning_image.shape)
    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = 1
        noise = next(noise_gen)[0]  # Noise batch of size (num_labels * samples_per_label, noise_dim)
    finally:
        noise_gen.batch_size = old_batch_size
    # Generate images from the generator: inputs are (conditioning_images, noise)
    generated_images = gen.predict([conditioning_image, noise])
    #print('shapes',generated_images.shape, cond.shape, noise.shape)
    return generated_images

def predict_images(gen, noise_gen, conditioning_image):
    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = conditioning_image.shape[0]
        noise = next(noise_gen)[0]  # Noise batch of size (num_labels * samples_per_label, noise_dim)
    finally:
        noise_gen.batch_size = old_batch_size
    # Generate images from the generator: inputs are (conditioning_images, noise)
    generated_images = gen.predict([conditioning_image, noise])
    #print('shapes',generated_images.shape, cond.shape, noise.shape)
    return generated_images


