from imagecorruptions import corrupt
from skimage import io
import numpy as np
from PIL import Image
import cv2
import random

def random_smooth_occlusion(image, num_shapes=1):
    # Ensure the image has an alpha channel
    # if image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Image dimensions
    height, width = image.shape[:2]

    for _ in range(num_shapes):
        # Random center point
        center = (random.randint(0, width), random.randint(0, height))
        # Random size
        axes = (random.randint(20, width//4), random.randint(20, height//4))
        # Random angle and color
        angle = random.randint(0, 180)
        # Random color and transparency
        color = (
            # random.randint(0, 255),
            # random.randint(0, 255),
            # random.randint(0, 255),
            # random.randint(50, 200)  # Alpha value, smaller means more transparent
            1, 1, 1, 0
        )
        # Draw the ellipse
        cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)

    return image

def add_salt_and_pepper_noise(image, severity=1):
    # Set noise ratio based on severity
    if severity < 1: severity = 1
    if severity > 5: severity = 5
    noise_ratio = severity * 0.02  # Noise ratio from 1% to 5%

    # Calculate the number of pixels to add noise to
    total_pixels = int(image.size * noise_ratio)
    rows, cols = image.shape[:2]

    for _ in range(total_pixels):
        # Randomly choose noise type: 0 for Pepper, 255 for Salt
        noise_type = random.choice([0, 255])
        # Randomly select pixel position
        x = random.randint(0, cols - 1)
        y = random.randint(0, rows - 1)
        # Apply noise
        if image.ndim == 2:  # Grayscale image
            image[y, x] = noise_type
        else:  # Color image
            image[y, x] = [noise_type]*3  # Apply the same noise to all channels

    return image

def random_hue_shift(image, severity=1):
    # Define maximum hue shift based on severity
    max_hue_shift = severity * 36  # Severity 1 corresponds to max 36 degrees, severity 5 corresponds to max 180 degrees

    # Randomly select a hue shift amount
    hue_shift = random.randint(-max_hue_shift, max_hue_shift)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Shift the hue channel of the HSV image
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

    # Convert the shifted HSV image back to BGR color space
    shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return shifted_image

def read_image(filepath):
    image = np.asarray(Image.open(filepath))
    # image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return image

def save_image(image, filepath):
    io.imsave(filepath, image.astype(np.uint8))

def pix_process(image, distortion_type, severity):
    # image = read_image(input_filepath)

    # Check if the image has 4 channels
    if image.shape[2] == 4:
        # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if distortion_type == "random_occlusion":
        # Apply random occlusion
        corrupted_image = random_smooth_occlusion(image, severity)
    elif distortion_type == "salt_and_pepper_noise":
        # Apply salt and pepper noise
        corrupted_image = add_salt_and_pepper_noise(image, severity)
    elif distortion_type == "random_hue_shift":
        # Apply random hue shift
        corrupted_image = random_hue_shift(image, severity)
    else:
        corrupted_image = corrupt(image, corruption_name=distortion_type, severity=severity)

    # return
    return corrupted_image

# distortion_type:
# gaussian_noise, impulse_noise, shot_noise, speckle_noise, salt_pepper_noise, defocus_blur, motion_blur, zoom_blur, frost_blur,
# spatter_blur, brightness, contrast, jpeg_compression, random_occlusion, color_shift, glass_blur, fog, snow, saturate, pixelate

if __name__ == "__main__":
    # Example usage:
    # input_filepath = 'A.jpeg'  # Path to the input image
    # distortion_type = 'snow'  # Type of distortion
    # severity = 1  # Severity of the distortion (1 to 5)

    # output_filepath = './out/output_image_'+distortion_type+'_'+str(severity)+'.png'  # Path to save the distorted image

    # pix_process(input_filepath, output_filepath, distortion_type, severity)
