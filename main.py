import math
import tkinter as tk

import numpy as np
from skimage.draw import line_nd
from sklearn.metrics import mean_squared_error



INTERMEDIATE_STEPS = 10

# 1D Convolution
def convolve_sinogram(img, kernelSize=9):
    # Convert to numpy array if not already
    sino = np.array(img)

    # Create kernel with the same logic as the original
    kernelCenter = int(kernelSize / 2)
    kernel = np.zeros(kernelSize)

    # Fill kernel with values
    for i in range(kernelSize):
        if i == kernelCenter:
            kernel[i] = 1.0
        elif i % 2 == 0:
            kernel[i] = 0.0
        else:
            kernel[i] = (-4 / pow(math.pi, 2)) / pow(i - kernelCenter, 2)

    # Create output array with same shape
    filtered_sino = np.zeros_like(sino)

    # Apply convolution to each row using numpy.convolve
    for i in range(sino.shape[0]):
        filtered_sino[i, :] = np.convolve(sino[i, :], kernel, mode='same')

    return filtered_sino


def get_max(sinogram):
    return max([item for k in sinogram for item in k])


def generate_parallel_rays(radius, center, angle=45, span=120, num_rays=20):
    alpha = math.radians(angle)
    theta = math.radians(span)
    ray_list = []

    for ray_idx in range(num_rays):
        x_detector = radius * math.cos(alpha - (ray_idx * theta / (num_rays - 1)) + theta / 2) + center[0]
        y_detector = radius * math.sin(alpha - (ray_idx * theta / (num_rays - 1)) + theta / 2) + center[1]
        x_source = radius * math.cos(alpha + math.pi - (theta / 2) + (ray_idx * theta / (num_rays - 1))) + center[0]
        y_source = radius * math.sin(alpha + math.pi - (theta / 2) + (ray_idx * theta / (num_rays - 1))) + center[1]

        ray_list.append([[x_detector, x_source], [y_detector, y_source]])

    return ray_list


def radon_transform(image, num_steps=60, span=120, num_rays=250, max_angle=180):
    sinogram = np.zeros((num_steps, num_rays))

    for step_idx in range(num_steps):
        angle = step_idx * (max_angle / num_steps)
        rays = generate_parallel_rays(max(image.shape[0] // 2, image.shape[1] // 2) * math.sqrt(2),
                                      (image.shape[0] // 2, image.shape[1] // 2), angle, span, num_rays)

        for ray_idx, ray in enumerate(rays):
            ray_value = 0

            # Get start/end points and convert to integers
            x_d, x_s = map(int, ray[0])
            y_d, y_s = map(int, ray[1])

            # Get line coordinates using skimage
            rr, cc = line_nd((y_d, x_d), (y_s, x_s), endpoint=True)

            # Accumulate values
            for r, c in zip(rr, cc):
                if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
                    ray_value += image[r][c]

            sinogram[step_idx][ray_idx] = ray_value

    return sinogram


def transpose_sinogram(sinogram):
    return [list(x) for x in zip(*sinogram)]


def normalize_data(data, max_value):
    return data / max_value


def inverse_radon_transform(sinogram, image, num_steps=60, span=120, num_rays=250,
                            max_angle=180, compute_error=False, save_intermediate=False):
    reconstructed_image = np.zeros(image.shape)
    error_list = []
    max_value = -1
    intermediate_images = []

    # Determine every how many steps to save intermediate images
    step_size = max(1, num_steps // INTERMEDIATE_STEPS)

    for step_idx in range(num_steps):
        angle = step_idx * max_angle / num_steps
        rays = generate_parallel_rays(max(image.shape[0] // 2, image.shape[1] // 2) * math.sqrt(2),
                                      (image.shape[0] // 2, image.shape[1] // 2), angle, span, num_rays)

        for ray_idx, ray in enumerate(rays):
            # Get start/end points and convert to integers
            x_d, x_s = map(int, ray[0])
            y_d, y_s = map(int, ray[1])

            # Get line coordinates using skimage
            rr, cc = line_nd((y_d, x_d), (y_s, x_s), endpoint=True)

            # Backproject values
            for r, c in zip(rr, cc):
                if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
                    reconstructed_image[r][c] += sinogram[step_idx][ray_idx]
                    max_value = max(max_value, reconstructed_image[r][c])

        if compute_error:
            error_list.append(mean_squared_error(reconstructed_image.flatten(),
                                                 image.flatten()))

        # Saving intermediate image
        if save_intermediate and ((step_idx + 1) % step_size == 0 or step_idx == num_steps - 1):
            norm_image = normalize_data(reconstructed_image.copy(), max_value)
            intermediate_images.append(norm_image)

    if not compute_error:
        if save_intermediate:
            return normalize_data(reconstructed_image, max_value), intermediate_images
        return normalize_data(reconstructed_image, max_value)

    if save_intermediate:
        return normalize_data(reconstructed_image, max_value), error_list, intermediate_images
    return normalize_data(reconstructed_image, max_value), error_list


def main():
    from ct_image_reconstruction_app import CTImageReconstructionApp
    root = tk.Tk()
    app = CTImageReconstructionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
