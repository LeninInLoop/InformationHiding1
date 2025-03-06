"""
Image Watermarking System
-------------------------
A modular system for embedding and extracting digital watermarks in images.
"""

import os
import os.path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Constants and Enums
class ImageChannels:
    """RGB color channel indices"""
    RED: int = 0
    GREEN: int = 1
    BLUE: int = 2


class WatermarkedImageGroup:
    """Watermark image types"""
    SUTECH: int = 0
    RANDOM: int = 1


# Image Generation and Manipulation
def generate_pseudo_random_image(shape: tuple, is_binary: bool) -> np.ndarray:
    """
    Generate a pseudo-random image of specified shape.

    Args:
        shape: Tuple of (height, width)
        is_binary: If True, return a binary image; otherwise grayscale

    Returns:
        Numpy array containing the generated image
    """
    if len(shape) != 2:
        raise ValueError('Image array must be two-dimensional')

    random_image = np.array(
        127.5 * np.random.randn(*shape) + 127.5,
        dtype=np.uint8
    )
    return random_image > 127.5 if is_binary else random_image


def generate_binary_image_from_gray_scale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to binary based on threshold.

    Args:
        image_array: 3D array representing an image

    Returns:
        Binary image as numpy array
    """
    return np.where(image_array[:, :, 0] > 127, 255, 0).astype(np.bool)


# File Operations
def save_image(image_array: np.ndarray, filename: str) -> None:
    """
    Save image array to file.

    Args:
        image_array: 2D or 3D array representing an image
        filename: Path to save the image
    """
    if len(image_array.shape) not in (2, 3):
        raise ValueError('Image array must be two or three-dimensional')

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Image.fromarray(image_array).save(filename)
    print(50 * "-", f"\nImage saved as {filename}")


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array
    """
    if not os.path.isfile(image_path):
        raise ValueError('Specified image file does not exist')

    return np.array(Image.open(image_path))


def resize_image(image_array: np.ndarray, target_dimensions: tuple[int, int]) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image_array: 2D or 3D array representing an image
        target_dimensions: Tuple of (width, height)

    Returns:
        Resized image as numpy array
    """
    if len(image_array.shape) not in (2, 3):
        raise ValueError('Image array must be two or three-dimensional')

    return np.array(Image.fromarray(image_array).resize(target_dimensions))


# Bit Plane Operations
def extract_binary_images(image_array: np.ndarray, channel_index: int) -> np.ndarray:
    """
    Extract bit planes from a specific channel of an image.

    Args:
        image_array: 3D array representing an image
        channel_index: Index of the channel to extract from

    Returns:
        3D array of bit planes
    """
    if len(image_array.shape) != 3:
        raise ValueError('Image array must be three-dimensional')

    binary_array = np.zeros(image_array[:, :, channel_index].shape + (8,))
    for bit in range(8):
        binary_array[:, :, bit] = np.bitwise_and(image_array[:, :, channel_index], 2 ** bit) > 0

    return np.array(binary_array, dtype=np.bool)


# Watermarking Operations
def embed_watermark(
        host_image: np.ndarray,
        watermark: np.ndarray,
        pixel_locations: List[Tuple[int, int]],
        bit_plane: int,
        channel: int
) -> np.ndarray:
    """
    Embed watermark into host image.

    Args:
        host_image: Image to embed watermark into
        watermark: Binary watermark image
        pixel_locations: List of (row, col) coordinates for embedding
        bit_plane: Bit plane to embed watermark (0-7)
        channel: Color channel to embed watermark

    Returns:
        Watermarked image as numpy array
    """
    host_copy = host_image.copy()
    selected_channel = host_copy[:, :, channel]

    # Create the bit mask to clear the target bit
    clear_mask = ~(1 << bit_plane) & 0xFF

    for idx, (i, j) in enumerate(pixel_locations):
        if i < host_image.shape[0] and j < host_image.shape[1] and idx < watermark.size:
            pixel_cleared = selected_channel[i, j] & clear_mask  # Clear bit
            selected_channel[i, j] = pixel_cleared | (watermark.flat[idx] << bit_plane)

    host_copy[:, :, channel] = selected_channel
    return host_copy


def extract_watermark(
        watermarked_image: np.ndarray,
        pixel_locations: List[Tuple[int, int]],
        bit_plane: int,
        channel: int,
        watermark_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract watermark from watermarked image.

    Args:
        watermarked_image: Image with embedded watermark
        pixel_locations: List of (row, col) coordinates for extraction
        bit_plane: Bit plane containing watermark (0-7)
        channel: Color channel containing watermark
        watermark_shape: Shape of watermark to extract

    Returns:
        Tuple of (extracted watermark, cleaned image)
    """
    clean_image = watermarked_image.copy()
    extracted_watermark = np.zeros(watermark_shape, dtype=np.bool)

    selected_channel = watermarked_image[:, :, channel]

    for idx, (i, j) in enumerate(pixel_locations):
        if idx < extracted_watermark.size:
            extracted_watermark.flat[idx] = (selected_channel[i, j] >> bit_plane) & 1  # Extract bit

    mask = 0xFF ^ (1 << bit_plane)  # Create mask with 0 at the bit_plane position
    for i, j in pixel_locations:
        if i < clean_image.shape[0] and j < clean_image.shape[1]:
            clean_image[i, j, channel] &= mask  # Clear the bit at bit_plane

    return extracted_watermark, clean_image


# Visualization Functions
def plot_bit_planes(binary_array: np.ndarray, filename: Optional[str] = None) -> None:
    """
    Plot the bit planes of an image.

    Args:
        binary_array: 3D array of bit planes
        filename: Optional path to save the plot
    """
    if len(binary_array.shape) != 3:
        raise ValueError('Input must be a 3D array')

    if binary_array.shape[2] != 8:
        raise ValueError('Input must contain 8 bit planes')

    plt.figure(figsize=(15, 8))

    # Plot each bit plane
    for bit in range(8):
        plt.subplot(2, 4, bit + 1)
        plt.imshow(binary_array[:, :, bit], cmap='gray')

        bit_label = (
            '(LSB)' if bit == 0
            else '(MSB)' if bit == 7
            else ''
        )
        plt.title(f'Bit Plane {bit} {bit_label}')
        plt.axis('off')

    plt.tight_layout()

    titles = (
        'Bit Planes from Least Significant Bit (LSB) '
        'to Most Significant Bit (MSB)'
    )
    plt.suptitle(
        titles,
        fontsize=16,
        y=1.02
    )

    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(50 * "-", f"\nImage saved as {filename}")
    plt.close()


def plot_watermarked_images(base_image_path: str, image_group: int, filename: str) -> None:
    """
    Plot watermarked images from all bit planes.

    Args:
        base_image_path: Base path to watermarked images
        image_group: Type of watermark (SUTECH or RANDOM)
        filename: Path to save the plot
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    image_channel_path = base_image_path.split(r"/")[-1]
    for bit in range(8):
        row, col = divmod(bit, 4)  # Arrange in 2 rows, 4 columns

        if image_group == WatermarkedImageGroup.SUTECH:
            # Load the image with SUTECH watermark
            image_channel = image_channel_path.split("SUTECH")[-1]
            image_path = base_image_path + image_channel + f'_host_image_with_SUTECH_watermark_bit_{bit}.tiff'
        else:
            # Load the image with RANDOM watermark
            image_channel = image_channel_path.split("Random")[-1]
            image_path = base_image_path + image_channel + f'_host_image_with_RANDOM_watermark_bit_{bit}.tiff'

        if os.path.exists(image_path):
            image = np.array(Image.open(image_path))
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'Bit {bit} - {"SUTECH" if image_group == 0 else "RANDOM"}')
            axes[row, col].axis('off')

    plt.tight_layout()

    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.close()


def plot_watermarked_and_clear_images(
        watermarked_image: np.ndarray,
        clean_image: np.ndarray,
        watermark: np.ndarray,
        filename: str
) -> None:
    """
    Create a comparison plot of watermarked image, extracted watermark, and clean image.

    Args:
        watermarked_image: Image with embedded watermark
        clean_image: Image with watermark removed
        watermark: Extracted watermark
        filename: Path to save the plot
    """
    # Set global font properties
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12
    })

    # Create a new figure with appropriate size
    fig = plt.figure(figsize=(16, 7))

    # Create a GridSpec layout with better proportions
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 2, 4], wspace=0.3)

    # Set figure background to white
    fig.patch.set_facecolor('white')

    # Create the axes for each image
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Remove ticks
        ax.tick_params(axis='both', which='both', length=0, colors='black')

    # Display watermarked image with improved contrast if needed
    ax1.imshow(watermarked_image)
    ax1.set_title('Watermarked Image', color='black', fontweight='bold', pad=20)
    ax1.axis('off')

    # Display extracted watermark with better color mapping
    ax2.imshow(watermark, cmap='gray', interpolation='nearest')
    ax2.set_title('Extracted\nWatermark', color='black', fontweight='bold', pad=20)
    ax2.axis('off')

    # Add a subtle border around the watermark
    rect = plt.Rectangle(
        (-0.5, -0.5),
        watermark.shape[1],
        watermark.shape[0],
        linewidth=2,
        edgecolor='gray',
        facecolor='none',
        alpha=0.5
    )
    ax2.add_patch(rect)

    # Display clean image
    ax3.imshow(clean_image)
    ax3.set_title('Clean Image', color='black', fontweight='bold', pad=20)
    ax3.axis('off')

    # Add a main title for the entire figure
    fig.suptitle('Image Watermarking Comparison',
                 color='black',
                 fontsize=16,
                 fontweight='bold',
                 y=0.98)

    # Add metadata text at the bottom
    metadata_text = f"Resolution: {watermarked_image.shape[1]}x{watermarked_image.shape[0]} px"
    fig.text(0.5, 0.02, metadata_text, ha='center', color='dimgray', fontsize=10)

    # Adjust spacing with more precision
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save with higher quality settings
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.2,
        transparent=False
    )
    plt.close()
    print(50 * "-", f"\nEnhanced comparison saved as {filename}")


# Directory Setup Function
def create_directory_structure() -> dict:
    """
    Create and return a directory structure for the watermarking process.

    Returns:
        Dictionary of directory paths
    """
    base_image_paths = {
        "root": os.path.join("Images"),
        "host": os.path.join("Images", "Host Image"),
        "bit_planes": os.path.join("Images", "Host Image", "Bit Planes"),
        # Algorithm 1 - Fixed Location
        "fixed_location": os.path.join("Images", "Fixed Location Algorithm"),
        "fixed_random": os.path.join("Images", "Fixed Location Algorithm", "Random"),
        "fixed_random_red": os.path.join("Images", "Fixed Location Algorithm", "Random", "RED"),
        "fixed_random_green": os.path.join("Images", "Fixed Location Algorithm", "Random", "GREEN"),
        "fixed_random_blue": os.path.join("Images", "Fixed Location Algorithm", "Random", "BLUE"),
        "fixed_sutech": os.path.join("Images", "Fixed Location Algorithm", "SUTECH"),
        "fixed_sutech_red": os.path.join("Images", "Fixed Location Algorithm", "SUTECH", "RED"),
        "fixed_sutech_green": os.path.join("Images", "Fixed Location Algorithm", "SUTECH", "GREEN"),
        "fixed_sutech_blue": os.path.join("Images", "Fixed Location Algorithm", "SUTECH", "BLUE"),
        # Algorithm 2 - Pseudo Random Location
        "pseudo_random": os.path.join("Images", "Pseudo Random Location Algorithm"),
        "pseudo_random_random": os.path.join("Images", "Pseudo Random Location Algorithm", "Random"),
        "pseudo_random_random_red": os.path.join("Images", "Pseudo Random Location Algorithm", "Random", "RED"),
        "pseudo_random_random_green": os.path.join("Images", "Pseudo Random Location Algorithm", "Random", "GREEN"),
        "pseudo_random_random_blue": os.path.join("Images", "Pseudo Random Location Algorithm", "Random", "BLUE"),
        "pseudo_random_sutech": os.path.join("Images", "Pseudo Random Location Algorithm", "SUTECH"),
        "pseudo_random_sutech_red": os.path.join("Images", "Pseudo Random Location Algorithm", "SUTECH", "RED"),
        "pseudo_random_sutech_green": os.path.join("Images", "Pseudo Random Location Algorithm", "SUTECH", "GREEN"),
        "pseudo_random_sutech_blue": os.path.join("Images", "Pseudo Random Location Algorithm", "SUTECH", "BLUE"),
    }

    # Ensure all directories exist
    for path in base_image_paths.values():
        os.makedirs(path, exist_ok=True)

    # Create extracted directories
    for algorithm in ['fixed', 'pseudo_random']:
        for watermark_type in ['SUTECH', 'Random']:
            # Create base directory for extracted comparisons
            folder_key = f"{algorithm}_{watermark_type.lower()}"
            if folder_key in base_image_paths:
                extracted_base = os.path.join(base_image_paths[folder_key], 'Extracted')
                os.makedirs(extracted_base, exist_ok=True)

                # Create channel-specific directories in Extracted
                for channel in ['RED', 'GREEN', 'BLUE']:
                    os.makedirs(os.path.join(extracted_base, channel), exist_ok=True)

    return base_image_paths


# Main Process Functions
def process_host_image(base_image_paths: dict) -> np.ndarray:
    """
    Load, resize and analyze the host image.

    Args:
        base_image_paths: Dictionary of directory paths

    Returns:
        Resized host image
    """
    # Load and resize host image
    host_image = load_image(os.path.join(base_image_paths['root'], 'host_image.jpeg'))
    resized_host_image = resize_image(host_image, target_dimensions=(512, 512))
    save_image(
        resized_host_image,
        filename=os.path.join(base_image_paths['host'], 'resized_host_image.tiff')
    )

    # Process each channel
    channel_configs = [
        (ImageChannels.RED, 'red_gray_scale.tiff', 'red_LSB_to_MSB_images.tiff'),
        (ImageChannels.GREEN, 'green_gray_scale.tiff', 'green_LSB_to_MSB_images.tiff'),
        (ImageChannels.BLUE, 'blue_gray_scale.tiff', 'blue_LSB_to_MSB_images.tiff')
    ]

    for channel, output_grayscale_filename, output_binary_filename in channel_configs:
        # Save grayscale channel
        save_image(
            resized_host_image[:, :, channel],
            filename=os.path.join(base_image_paths['host'], output_grayscale_filename)
        )

        # Extract and plot bit planes
        binary_images = extract_binary_images(resized_host_image, channel_index=channel)
        plot_bit_planes(
            binary_array=binary_images,
            filename=os.path.join(base_image_paths['bit_planes'], output_binary_filename)
        )

    return resized_host_image


def process_watermarks(base_image_paths: dict, host_image: np.ndarray) -> tuple:
    """
    Process and embed watermarks into the host image.

    Args:
        base_image_paths: Dictionary of directory paths
        host_image: Host image to embed watermarks into

    Returns:
        Tuple of (binary_watermark, pseudo_random_image, pixel_locations)
    """
    # Create pseudo-random watermark
    pseudo_random_image = generate_pseudo_random_image(shape=(64, 64), is_binary=True)
    save_image(
        pseudo_random_image,
        filename=os.path.join(base_image_paths['root'], 'pseudo_random_image.tiff')
    )

    # Load and process SUTECH watermark
    watermark = load_image(os.path.join(base_image_paths['root'], 'sutech.jpg'))
    print(50 * "-", "\nWatermark shape:", watermark.shape)

    binary_watermark = generate_binary_image_from_gray_scale(watermark)
    save_image(
        image_array=binary_watermark,
        filename=os.path.join(base_image_paths['root'], 'binary_sutech_watermark.tiff')
    )
    print(50 * "-", "\nBinary watermark:\n", binary_watermark)

    # Define fixed locations based on watermark size
    pixel_locations = [(i, j) for i in range(binary_watermark.shape[0]) for j in range(binary_watermark.shape[1])]

    # Channel configurations
    channels = {
        ImageChannels.RED: {'base_address': "red", 'save_address': "RED_host_image_with_"},
        ImageChannels.GREEN: {'base_address': "green", 'save_address': "GREEN_host_image_with_"},
        ImageChannels.BLUE: {'base_address': "blue", 'save_address': "BLUE_host_image_with_"}
    }

    # Embed watermarks in each bit plane and channel
    for bit in range(8):
        for channel, address in channels.items():
            # Embed SUTECH watermark
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=binary_watermark,
                pixel_locations=pixel_locations,
                bit_plane=bit,
                channel=channel,
            )
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['fixed_sutech_' + address['base_address']],
                    address["save_address"] + f'SUTECH_watermark_bit_{bit}.tiff'
                )
            )

            # Embed RANDOM watermark
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=pseudo_random_image,
                pixel_locations=pixel_locations,
                bit_plane=bit,
                channel=channel
            )
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['fixed_random_' + address['base_address']],
                    address["save_address"] + f'RANDOM_watermark_bit_{bit}.tiff'
                )
            )

    # Generate summary plots for each channel
    for address in channels.values():
        plot_watermarked_images(
            base_image_path=base_image_paths['fixed_sutech_' + address['base_address']],
            image_group=WatermarkedImageGroup.SUTECH,
            filename=os.path.join(base_image_paths['fixed_sutech'], address["save_address"] + "SUTECH_watermarks.tiff")
        )
        plot_watermarked_images(
            base_image_path=base_image_paths['fixed_random_' + address['base_address']],
            image_group=WatermarkedImageGroup.RANDOM,
            filename=os.path.join(base_image_paths['fixed_random'], address["save_address"] + "RANDOM_watermarks.tiff")
        )

    return binary_watermark, pseudo_random_image, pixel_locations


def extract_all_watermarks(
        base_image_paths: dict,
        binary_watermark: np.ndarray,
        pseudo_random_image: np.ndarray,
        pixel_locations: List[Tuple[int, int]]
) -> None:
    """
    Extract and compare watermarks from all watermarked images.

    Args:
        base_image_paths: Dictionary of directory paths
        binary_watermark: SUTECH binary watermark
        pseudo_random_image: Random binary watermark
        pixel_locations: List of pixel locations for extraction
    """
    print(50 * "-", "\nExtracting watermarks from all images...")

    # Process fixed location algorithm
    for channel_name, channel_val in {
        'RED': ImageChannels.RED,
        'GREEN': ImageChannels.GREEN,
        'BLUE': ImageChannels.BLUE
    }.items():
        for bit in range(8):
            # Process SUTECH watermarks
            sutech_img_path = os.path.join(
                base_image_paths[f'fixed_sutech_{channel_name.lower()}'],
                f'{channel_name}_host_image_with_SUTECH_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(sutech_img_path):
                watermarked_image = load_image(sutech_img_path)
                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=pixel_locations,
                    bit_plane=bit,
                    channel=channel_val,
                    watermark_shape=binary_watermark.shape
                )

                # Save comparison image
                comparison_path = os.path.join(
                    base_image_paths['fixed_sutech'],
                    'Extracted',
                    channel_name,
                    f'comparison_SUTECH_{channel_name}_bit_{bit}.png'
                )
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=comparison_path
                )

            # Process RANDOM watermarks
            random_img_path = os.path.join(
                base_image_paths[f'fixed_random_{channel_name.lower()}'],
                f'{channel_name}_host_image_with_RANDOM_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(random_img_path):
                watermarked_image = load_image(random_img_path)
                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=pixel_locations,
                    bit_plane=bit,
                    channel=channel_val,
                    watermark_shape=pseudo_random_image.shape
                )

                # Save comparison image
                comparison_path = os.path.join(
                    base_image_paths['fixed_random'],
                    'Extracted',
                    channel_name,
                    f'comparison_RANDOM_{channel_name}_bit_{bit}.png'
                )
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=comparison_path
                )

    print(50 * "-", "\nWatermark extraction and comparison complete!")


def main():
    """Main function to run the watermarking process."""
    # Create directory structure
    base_image_paths = create_directory_structure()

    # Process host image
    host_image = process_host_image(base_image_paths)

    # Process and embed watermarks
    binary_watermark, pseudo_random_image, pixel_locations = process_watermarks(
        base_image_paths,
        host_image
    )

    # Extract and compare watermarks
    extract_all_watermarks(
        base_image_paths,
        binary_watermark,
        pseudo_random_image,
        pixel_locations
    )


if __name__ == "__main__":
    main()