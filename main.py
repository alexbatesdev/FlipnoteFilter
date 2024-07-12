# This code was written by an AI assistant
import numpy as np
from PIL import Image, ImageEnhance

def closest_color(pixel, palette):
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    return palette[np.argmin(distances)]

def color_quantization(image, palette):
    data = np.array(image)
    # Flatten the image array and find the closest colors
    flat_data = data.reshape((-1, 3))
    quantized_data = np.array([closest_color(pixel, palette) for pixel in flat_data])
    quantized_data = quantized_data.reshape(data.shape)
    return Image.fromarray(quantized_data.astype(np.uint8))

def dithering(image):
    data = np.array(image.convert('L'), dtype=np.float32)  # Convert to grayscale
    threshold_matrix = np.array([
        [0, 128],
        [192, 64]
    ], dtype=np.float32)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            old_pixel = data[i, j]
            new_pixel = 255 * (old_pixel > threshold_matrix[i % 2, j % 2])
            data[i, j] = new_pixel
            error = old_pixel - new_pixel
            if j + 1 < data.shape[1]:
                data[i, j + 1] += error * 7 / 16
            if i + 1 < data.shape[0]:
                if j > 0:
                    data[i + 1, j - 1] += error * 3 / 16
                data[i + 1, j] += error * 5 / 16
                if j + 1 < data.shape[1]:
                    data[i + 1, j + 1] += error * 1 / 16

    return Image.fromarray(data.astype(np.uint8))

def color_dithering(image, palette, pattern):
    data = np.array(image, dtype=np.float32)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            old_pixel = data[y, x]
            new_pixel = closest_color(old_pixel, palette)
            data[y, x] = new_pixel
            error = old_pixel - new_pixel
            for dy, dx, factor in pattern:
                ny, nx = y + dy, x + dx
                if 0 <= nx < data.shape[1] and 0 <= ny < data.shape[0]:
                    data[ny, nx] += error * factor
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))

def get_bayer_matrix(level):
    bayer2 = np.array([[0, 2], [3, 1]]) / 4.0 - 0.5
    bayer4 = (np.array([[0, 8, 2, 10],
                        [12, 4, 14, 6],
                        [3, 11, 1, 9],
                        [15, 7, 13, 5]]) / 16.0 - 0.5)
    bayer8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                        [48, 16, 56, 24, 50, 18, 58, 26],
                        [12, 44, 4, 36, 14, 46, 6, 38],
                        [60, 28, 52, 20, 62, 30, 54, 22],
                        [3, 35, 11, 43, 1, 33, 9, 41],
                        [51, 19, 59, 27, 49, 17, 57, 25],
                        [15, 47, 7, 39, 13, 45, 5, 37],
                        [63, 31, 55, 23, 61, 29, 53, 21]]) / 64.0 - 0.5)
    return [bayer2, bayer4, bayer8][level]

def get_ordered_matrix(size):
    ordered2 = np.array([[1, 3], [4, 2]]) / 5.0 - 0.5
    ordered4 = (np.array([[1, 9, 3, 11],
                          [13, 5, 15, 7],
                          [4, 12, 2, 10],
                          [16, 8, 14, 6]]) / 17.0 - 0.5)
    ordered8 = (np.array([[1, 33, 9, 41, 3, 35, 11, 43],
                          [49, 17, 57, 25, 51, 19, 59, 27],
                          [13, 45, 5, 37, 15, 47, 7, 39],
                          [61, 29, 53, 21, 63, 31, 55, 23],
                          [4, 36, 12, 44, 2, 34, 10, 42],
                          [52, 20, 60, 28, 50, 18, 58, 26],
                          [16, 48, 8, 40, 14, 46, 6, 38],
                          [64, 32, 56, 24, 62, 30, 54, 22]]) / 65.0 - 0.5)
    return {2: ordered2, 4: ordered4, 8: ordered8}[size]

def get_void_and_cluster_matrix(size):
    void_cluster4 = np.array([[0, 0.6, 0.2, 0.8],
                              [0.9, 0.3, 0.7, 0.1],
                              [0.4, 1.0, 0.5, 0.1],
                              [0.6, 0.0, 0.8, 0.2]]) - 0.5
    return {4: void_cluster4}[size]

def apply_threshold_dithering(image, threshold=128):
    img_array = np.array(image)
    gray_image = np.mean(img_array, axis=2)
    dithered_image = (gray_image > threshold) * 255
    return Image.fromarray(dithered_image.astype(np.uint8))

def apply_dithering(image, spread=0.5, red_color_count=2, green_color_count=2, blue_color_count=2, pattern='bayer', level=0):
    if pattern == 'bayer':
        dither_matrix = get_bayer_matrix(level)
    elif pattern == 'ordered':
        dither_matrix = get_ordered_matrix(2 ** (level + 1))
    elif pattern == 'void-cluster':
        dither_matrix = get_void_and_cluster_matrix(4)
    elif pattern == 'threshold':
        return apply_threshold_dithering(image)
    else:
        raise ValueError("Unsupported pattern. Choose 'bayer', 'ordered', 'void-cluster', or 'threshold'.")


    img_array = np.array(image)
    height, width, _ = img_array.shape

    for y in range(height):
        for x in range(width):
            matrix_value = dither_matrix[y % dither_matrix.shape[0], x % dither_matrix.shape[1]]
            pixel = img_array[y, x].astype(float)
            pixel += spread * matrix_value * 255.0

            pixel[0] = np.floor((red_color_count - 1) * pixel[0] / 255.0 + 0.5) * (255.0 / (red_color_count - 1))
            pixel[1] = np.floor((green_color_count - 1) * pixel[1] / 255.0 + 0.5) * (255.0 / (green_color_count - 1))
            pixel[2] = np.floor((blue_color_count - 1) * pixel[2] / 255.0 + 0.5) * (255.0 / (blue_color_count - 1))

            img_array[y, x] = np.clip(pixel, 0, 255)

    dithered_image = Image.fromarray(img_array.astype(np.uint8))
    return dithered_image


def scale_image(image, scale_multiplier, resample_filter=Image.NEAREST):
    width, height = image.size
    new_width = int(width * 2**scale_multiplier)
    new_height = int(height * 2**scale_multiplier)   

    return image.resize((new_width, new_height), resample_filter)

def downscale_image(image, target_width, target_height, resample_filter):
    return image.resize((target_width, target_height), resample_filter)

# Define the palette (R, G, B) for Red, Green, Blue, Yellow, Black, White
palette = np.array([
    [255, 16, 16],    # Red
    [0, 134, 49],    # Green
    [0, 56, 206],    # Blue
    [255, 231, 0],  # Yellow
    [0, 0, 0],      # Black
    [255, 255, 255] # White
])

floyd_steinberg_pattern = [
    (0, 1, 7/16),
    (1, -1, 3/16),
    (1, 0, 5/16),
    (1, 1, 1/16)
]

# Load image
image_path = input('Enter the path to the image: ')
image = Image.open(image_path)

image_name = image_path.split('.')[0].split('/')[-1]

brightened_image = ImageEnhance.Brightness(image).enhance(1.5)

downscaled_image = scale_image(image, -4)
downscaled_image.save(f'./output/downscaled_{image_name}.png')

# # Apply color quantization
quantized_image = color_quantization(downscaled_image, palette)
quantized_image.save(f'./output/quantized_{image_name}.png')

# Apply color dithering
colored_dithered_image = color_dithering(downscaled_image, palette, floyd_steinberg_pattern)
colored_dithered_image.save(f'./output/colored_dithered_{image_name}.png')

colored_dithered_image = apply_dithering(downscaled_image)
colored_dithered_image.save(f'./output/acerola_dithered_0_{image_name}.png')
colored_dithered_image = apply_dithering(downscaled_image, level=1)
colored_dithered_image.save(f'./output/acerola_dithered_1_{image_name}.png')
colored_dithered_image = apply_dithering(downscaled_image, level=2)
colored_dithered_image.save(f'./output/acerola_dithered_2_{image_name}.png')

colored_dithered_image = apply_dithering(downscaled_image, pattern='ordered')
colored_dithered_image.save(f'./output/ordered_dithered_{image_name}.png')

# # Apply dithering
# dithered_image = dithering(image)
# dithered_image.save('dithered_image.jpg')
# # This code was written by an AI assistant
