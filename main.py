import os
import os.path
from typing import Optional, List, Tuple
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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


class WatermarkAlgorithm:
    """Watermarking algorithm types"""
    FIXED_LOCATION: int = 0
    PSEUDO_RANDOM_LOCATION: int = 1


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
        raise ValueError('shape must be two-dimensional')

    random_image = np.array(
        52.05 * np.random.randn(*shape) + 127.5,
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
    return np.array(Image.fromarray(image_array).convert('1'))


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


# Random Location Generation
def generate_random_pixel_locations(
        host_image_shape: Tuple[int, int, int],
        watermark_size: int,
        seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Generate pseudo-random pixel locations for watermark embedding.

    Args:
        host_image_shape: Shape of host image (height, width, channels)
        watermark_size: Number of pixels in the watermark
        seed: Random seed for reproducibility

    Returns:
        List of (row, col) coordinates for embedding
    """
    # Set random seed for reproducibility
    random.seed(seed)

    height, width = host_image_shape[0], host_image_shape[1]
    
    # Generate unique random locations
    all_pixels = [(i, j) for i in range(height) for j in range(width)]
    
    if watermark_size > len(all_pixels):
        raise ValueError("Watermark size exceeds available pixels in host image")
    
    # Randomly select pixels
    random_locations = random.sample(all_pixels, watermark_size)

    return random_locations


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


# Save and load location map
def save_location_map(pixel_locations: List[Tuple[int, int]], filename: str) -> None:
    """
    Save pixel locations to a file for later retrieval.

    Args:
        pixel_locations: List of (row, col) coordinates
        filename: Path to save the location map
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to numpy array for efficient storage
    location_array = np.array(pixel_locations)
    np.save(filename, location_array)
    print(50 * "-", f"\n\033[91mLocation map saved as {filename}\033[0m")


def load_location_map(filename: str) -> List[Tuple[int, int]]:
    """
    Load pixel locations from a file.

    Args:
        filename: Path to the location map file

    Returns:
        List of (row, col) coordinates
    """
    if not os.path.isfile(filename):
        raise ValueError('Specified location map file does not exist')
    
    location_array = np.load(filename)
    return [(int(row), int(col)) for row, col in location_array]


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


def visualize_random_locations(
        host_image: np.ndarray,
        pixel_locations: List[Tuple[int, int]],
        filename: str
) -> None:
    """
    Create a visualization of random pixel locations used for watermark embedding.

    Args:
        host_image: Original host image
        pixel_locations: List of (row, col) coordinates used for embedding
        filename: Path to save the visualization
    """
    # Create a copy of the host image
    visualization = host_image.copy()

    # Create a mask image with the same dimensions
    mask = np.zeros(host_image.shape[:2], dtype=np.bool)

    # Mark the random locations
    for i, j in pixel_locations:
        if i < host_image.shape[0] and j < host_image.shape[1]:
            mask[i, j] = True

    # Create a highlighted visualization
    # Make a translucent red overlay for the random pixels
    for c in range(3):
        channel = visualization[:, :, c]
        if c == 0:  # Red channel
            channel[mask] = np.minimum(channel[mask] + 100, 255)  # Brighten red
        else:  # Green and Blue channels
            channel[mask] = np.maximum(channel[mask] - 50, 0)  # Darken others

    # Set up the figure
    plt.figure(figsize=(10, 10))
    plt.imshow(visualization)
    plt.title('Random Watermark Embedding Locations', fontsize=16)
    plt.axis('off')

    # Calculate and display percentage of image used
    pixel_count = len(pixel_locations)
    total_pixels = host_image.shape[0] * host_image.shape[1]
    used_percentage = (pixel_count / total_pixels) * 100

    plt.figtext(
        0.5, 0.01,
        f'Watermark uses {pixel_count} random pixels ({used_percentage:.2f}% of image)',
        ha='center',
        fontsize=12
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the visualization
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    print(50 * "-", f"\nRandom locations visualization saved as {filename}")


# Directory Setup Function
def create_directory_structure() -> dict:
    """
    Create and return a directory structure for the watermarking process.

    Returns:
        Dictionary of directory paths
    """
    base_image_paths = {
        "root": os.path.join("Images"),
        "evaluation": os.path.join("Images", "Evaluation"),
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
        # Location Maps for Random Algorithm
        "location_maps": os.path.join("Images", "Pseudo Random Location Algorithm" ,"Location Maps"),
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
        Tuple of (binary_watermark, pseudo_random_image, fixed_locations, random_locations_sutech, random_locations_random)
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

    # Define fixed locations based on watermark size (for Algorithm 1)
    fixed_locations = [(i, j) for i in range(binary_watermark.shape[0]) for j in range(binary_watermark.shape[1])]

    # Generate random locations for SUTECH watermark (for Algorithm 2)
    random_locations_sutech = generate_random_pixel_locations(
        host_image_shape=host_image.shape,
        watermark_size=binary_watermark.size,
        seed=42  # Use a fixed seed for reproducibility
    )
    # Save the location map for later extraction
    save_location_map(
        pixel_locations=random_locations_sutech,
        filename=os.path.join(base_image_paths['location_maps'], 'sutech_random_locations.npy')
    )
    
    # Generate random locations for RANDOM watermark (for Algorithm 2)
    random_locations_random = generate_random_pixel_locations(
        host_image_shape=host_image.shape,
        watermark_size=pseudo_random_image.size,
        seed=43  # Use a different seed
    )
    # Save the location map for later extraction
    save_location_map(
        pixel_locations=random_locations_random,
        filename=os.path.join(base_image_paths['location_maps'], 'random_random_locations.npy')
    )
    
    # Visualize the random locations
    visualize_random_locations(
        host_image=host_image,
        pixel_locations=random_locations_sutech,
        filename=os.path.join(base_image_paths['location_maps'], 'sutech_random_locations_visualization.png')
    )
    visualize_random_locations(
        host_image=host_image,
        pixel_locations=random_locations_random,
        filename=os.path.join(base_image_paths['location_maps'], 'random_random_locations_visualization.png')
    )

    # Channel configurations
    channels = {
        ImageChannels.RED: {'base_address': "red", 'save_address': "RED_host_image_with_"},
        ImageChannels.GREEN: {'base_address': "green", 'save_address': "GREEN_host_image_with_"},
        ImageChannels.BLUE: {'base_address': "blue", 'save_address': "BLUE_host_image_with_"}
    }

    # -------------------- Algorithm 1: Fixed Location Embedding --------------------
    print(50 * "-", "\n\033[91mRunning Algorithm 1: Fixed Location Embedding...\033[0m")
    
    # Embed watermarks in each bit plane and channel
    for bit in range(8):
        for channel, address in channels.items():
            # Embed SUTECH watermark
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=binary_watermark,
                pixel_locations=fixed_locations,
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
                pixel_locations=fixed_locations,
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

    # -------------------- Algorithm 2: Pseudo-Random Location Embedding --------------------
    print(50 * "-", "\n\033[91mRunning Algorithm 2: Pseudo-Random Location Embedding...\033[0m")
    
    # Embed watermarks in each bit plane and channel with random locations
    for bit in range(8):
        for channel, address in channels.items():
            # Embed SUTECH watermark with random locations
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=binary_watermark,
                pixel_locations=random_locations_sutech,
                bit_plane=bit,
                channel=channel,
            )
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['pseudo_random_sutech_' + address['base_address']],
                    address["save_address"] + f'SUTECH_watermark_bit_{bit}.tiff'
                )
            )
            # Embed RANDOM watermark
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=pseudo_random_image,
                pixel_locations=random_locations_random,
                bit_plane=bit,
                channel=channel
            )
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['pseudo_random_random_' + address['base_address']],
                    address["save_address"] + f'RANDOM_watermark_bit_{bit}.tiff'
                )
            )
        # Generate summary plots for each channel with random locations
        for address in channels.values():
            plot_watermarked_images(
                base_image_path=base_image_paths['pseudo_random_sutech_' + address['base_address']],
                image_group=WatermarkedImageGroup.SUTECH,
                filename=os.path.join(base_image_paths['pseudo_random_sutech'],
                                      address["save_address"] + "SUTECH_watermarks.tiff")
            )
            plot_watermarked_images(
                base_image_path=base_image_paths['pseudo_random_random_' + address['base_address']],
                image_group=WatermarkedImageGroup.RANDOM,
                filename=os.path.join(base_image_paths['pseudo_random_random'],
                                      address["save_address"] + "RANDOM_watermarks.tiff")
            )

    return binary_watermark, pseudo_random_image, fixed_locations, random_locations_sutech, random_locations_random


def extract_watermarks(
        base_image_paths: dict,
        host_image: np.ndarray,
        binary_watermark: np.ndarray,
        pseudo_random_image: np.ndarray,
        fixed_locations: List[Tuple[int, int]],
        random_locations_sutech: List[Tuple[int, int]],
        random_locations_random: List[Tuple[int, int]]
) -> None:
    """
    Extract and visualize watermarks from watermarked images.

    Args:
        base_image_paths: Dictionary of directory paths
        host_image: Original host image
        binary_watermark: SUTECH binary watermark
        pseudo_random_image: Random binary watermark
        fixed_locations: Fixed pixel locations for Algorithm 1
        random_locations_sutech: Random pixel locations for SUTECH watermark
        random_locations_random: Random pixel locations for random watermark
    """
    # Channel configurations
    channels = {
        ImageChannels.RED: {'base_address': "red", 'save_address': "RED_host_image_with_"},
        ImageChannels.GREEN: {'base_address': "green", 'save_address': "GREEN_host_image_with_"},
        ImageChannels.BLUE: {'base_address': "blue", 'save_address': "BLUE_host_image_with_"}
    }

    # -------------------- Algorithm 1: Extract Fixed Location Watermarks --------------------
    print(50 * "-", "\n\033[91mExtracting Algorithm 1: Fixed Location Watermarks...\033[0m")

    for bit in range(8):
        for channel, address in channels.items():
            # Extract SUTECH watermark
            watermarked_image_path = os.path.join(
                base_image_paths['fixed_sutech_' + address['base_address']],
                address["save_address"] + f'SUTECH_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(watermarked_image_path):
                watermarked_image = load_image(watermarked_image_path)

                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=fixed_locations,
                    bit_plane=bit,
                    channel=channel,
                    watermark_shape=binary_watermark.shape
                )

                # Save and visualize the extracted watermark
                save_image(
                    image_array=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['fixed_sutech'], 'Extracted', address['base_address'].upper(),
                        f'extracted_SUTECH_watermark_bit_{bit}.tiff'
                    )
                )
                save_image(
                    image_array=clean_image,
                    filename=os.path.join(
                        base_image_paths['fixed_sutech'], 'Extracted', address['base_address'].upper(),
                        f'extracted_SUTECH_clean_image_bit_{bit}.tiff'
                    )
                )
                # Create comparison visualization
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['fixed_sutech'], 'Extracted', address['base_address'].upper(),
                        f'comparison_SUTECH_bit_{bit}.png'
                    )
                )

            # Extract RANDOM watermark
            watermarked_image_path = os.path.join(
                base_image_paths['fixed_random_' + address['base_address']],
                address["save_address"] + f'RANDOM_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(watermarked_image_path):
                watermarked_image = load_image(watermarked_image_path)

                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=fixed_locations,
                    bit_plane=bit,
                    channel=channel,
                    watermark_shape=pseudo_random_image.shape
                )

                # # Save and visualize the extracted watermark
                save_image(
                    image_array=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['fixed_random'], 'Extracted', address['base_address'].upper(),
                        f'extracted_RANDOM_watermark_bit_{bit}.tiff'
                    )
                )
                save_image(
                    image_array=clean_image,
                    filename=os.path.join(
                        base_image_paths['fixed_random'], 'Extracted', address['base_address'].upper(),
                        f'extracted_RANDOM_clean_image_bit_{bit}.tiff'
                    )
                )
                # Create comparison visualization
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['fixed_random'], 'Extracted', address['base_address'].upper(),
                        f'comparison_RANDOM_bit_{bit}.png'
                    )
                )

    # -------------------- Algorithm 2: Extract Pseudo-Random Location Watermarks --------------------
    print(50 * "-", "\n\033[91mExtracting Algorithm 2: Pseudo-Random Location Watermarks...\033[0m")

    for bit in range(8):
        for channel, address in channels.items():
            # Extract SUTECH watermark with random locations
            watermarked_image_path = os.path.join(
                base_image_paths['pseudo_random_sutech_' + address['base_address']],
                address["save_address"] + f'SUTECH_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(watermarked_image_path):
                watermarked_image = load_image(watermarked_image_path)

                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=random_locations_sutech,
                    bit_plane=bit,
                    channel=channel,
                    watermark_shape=binary_watermark.shape
                )

                # # Save and visualize the extracted watermark
                save_image(
                    image_array=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_sutech'], 'Extracted', address['base_address'].upper(),
                        f'extracted_SUTECH_watermark_bit_{bit}.tiff'
                    )
                )
                save_image(
                    image_array=clean_image,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_sutech'], 'Extracted', address['base_address'].upper(),
                        f'extracted_SUTECH_clean_image_bit_{bit}.tiff'
                    )
                )
                # Create comparison visualization
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_sutech'], 'Extracted', address['base_address'].upper(),
                        f'comparison_SUTECH_bit_{bit}.png'
                    )
                )

            # Extract RANDOM watermark with random locations
            watermarked_image_path = os.path.join(
                base_image_paths['pseudo_random_random_' + address['base_address']],
                address["save_address"] + f'RANDOM_watermark_bit_{bit}.tiff'
            )

            if os.path.exists(watermarked_image_path):
                watermarked_image = load_image(watermarked_image_path)

                extracted_watermark, clean_image = extract_watermark(
                    watermarked_image=watermarked_image,
                    pixel_locations=random_locations_random,
                    bit_plane=bit,
                    channel=channel,
                    watermark_shape=pseudo_random_image.shape
                )

                # Save and visualize the extracted watermark
                save_image(
                    image_array=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_random'], 'Extracted', address['base_address'].upper(),
                        f'extracted_RANDOM_watermark_bit_{bit}.tiff'
                    )
                )

                save_image(
                    image_array=clean_image,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_random'], 'Extracted', address['base_address'].upper(),
                        f'extracted_RANDOM_clean_image_bit_{bit}.tiff'
                    )
                )

                # Create comparison visualization
                plot_watermarked_and_clear_images(
                    watermarked_image=watermarked_image,
                    clean_image=clean_image,
                    watermark=extracted_watermark,
                    filename=os.path.join(
                        base_image_paths['pseudo_random_random'], 'Extracted', address['base_address'].upper(),
                        f'comparison_RANDOM_bit_{bit}.png'
                    )
                )


def evaluate_watermarks(
        base_image_paths: dict,
        binary_watermark: np.ndarray,
        pseudo_random_image: np.ndarray
) -> None:
    """
    Evaluate watermarking effectiveness by analyzing extraction results.

    Args:
        base_image_paths: Dictionary of directory paths
        binary_watermark: SUTECH binary watermark
        pseudo_random_image: Random binary watermark
    """
    # Define channels and bits to evaluate
    channels = ['RED', 'GREEN', 'BLUE']
    bits = range(8)  # LSB, middle bit, MSB

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Algorithm and watermark types
    algorithms = ['fixed', 'pseudo_random']
    watermark_types = ['sutech', 'random']

    # Titles for the grid
    subplot_titles = [
        'Fixed Location - SUTECH Watermark',
        'Fixed Location - Random Watermark',
        'Random Location - SUTECH Watermark',
        'Random Location - Random Watermark'
    ]

    # Set titles for the grid
    for i, title in enumerate(subplot_titles):
        row, col = i // 2, i % 2
        axs[row, col].set_title(title)

    # Calculate similarity metrics for each combination
    for alg_idx, algorithm in enumerate(algorithms):
        for wm_idx, watermark_type in enumerate(watermark_types):
            reference_watermark = binary_watermark if watermark_type == 'sutech' else pseudo_random_image

            # Dictionary to store channel data for plotting
            channel_data = {channel: {'bits': [], 'similarities': []} for channel in channels}

            # Process each channel and bit combination
            for channel in channels:
                for bit in bits:
                    # Define path to extracted watermark
                    extracted_path = os.path.join(
                        base_image_paths[f'{algorithm}_{watermark_type}'],
                        'Extracted',
                        channel,
                        f'extracted_{"SUTECH" if watermark_type == "sutech" else "RANDOM"}_watermark_bit_{bit}.tiff'
                    )

                    # Check if file exists before attempting to load
                    if os.path.exists(extracted_path):
                        try:
                            # Load the extracted watermark
                            extracted_watermark = load_image(extracted_path)

                            # Calculate similarity (percentage of matching pixels)
                            if extracted_watermark.shape == reference_watermark.shape:
                                similarity = np.mean(extracted_watermark == reference_watermark) * 100
                            else:
                                # Reshape if needed
                                extracted_flat = extracted_watermark.flatten()
                                reference_flat = reference_watermark.flatten()
                                min_size = min(extracted_flat.size, reference_flat.size)
                                similarity = np.mean(extracted_flat[:min_size] == reference_flat[:min_size]) * 100

                            # Store the data for plotting
                            channel_data[channel]['bits'].append(bit)
                            channel_data[channel]['similarities'].append(similarity)

                        except Exception as e:
                            print(f"Error processing {extracted_path}: {e}")

            # Plot the results for each channel
            has_data = False
            for channel in channels:
                bits_list = channel_data[channel]['bits']
                sim_values = channel_data[channel]['similarities']

                # Only plot if we have data
                if bits_list and sim_values:
                    has_data = True
                    axs[alg_idx, wm_idx].plot(bits_list, sim_values, marker='o', linestyle='-',
                                              label=channel, linewidth=2)

            # Set axis labels and formatting
            axs[alg_idx, wm_idx].set_xlabel('Bit Plane')
            axs[alg_idx, wm_idx].set_ylabel('Similarity (%)')
            axs[alg_idx, wm_idx].set_ylim(0, 105)

            # Only add legend if we have data
            if has_data:
                axs[alg_idx, wm_idx].legend()
            else:
                axs[alg_idx, wm_idx].text(0.5, 0.5, 'No data available',
                                          ha='center', va='center', transform=axs[alg_idx, wm_idx].transAxes)

            # Add grid
            axs[alg_idx, wm_idx].grid(True, linestyle='--', alpha=0.7)

            # Set x-ticks to only show the bit values we're using
            axs[alg_idx, wm_idx].set_xticks(bits)

    # Add a main title
    fig.suptitle('Watermark Extraction Similarity Analysis', fontsize=16, y=0.98)

    # Adjust layout without using tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    # Save figure
    output_path = os.path.join(base_image_paths['evaluation'], 'watermark_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{50 * '-'}\n\033[91mWatermark evaluation saved to {output_path}\033[0m")


def evaluate_host_image(
        base_image_paths: dict,
        host_image: np.ndarray,
) -> None:
    """
    Evaluate and plot the comparison between host images and clean images after watermark extraction.
    Uses both PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) as quality metrics.

    Args:
        base_image_paths: Dictionary of directory paths
        host_image: Original host image
    """
    # Import SSIM
    from skimage.metrics import structural_similarity as ssim

    # Define channels and bits to evaluate
    channels = ['RED', 'GREEN', 'BLUE']
    bits = range(8)

    # Algorithm and watermark types
    algorithms = ['fixed', 'pseudo_random']
    watermark_types = ['sutech', 'random']

    # Create figures with subplots - one for PSNR and one for SSIM
    fig_metrics, axs_metrics = plt.subplots(2, 2, figsize=(18, 12))

    # Titles for the grid
    subplot_titles = [
        'Fixed Location - SUTECH Watermark',
        'Fixed Location - Random Watermark',
        'Random Location - SUTECH Watermark',
        'Random Location - Random Watermark'
    ]

    # Set titles for the grid
    for i, title in enumerate(subplot_titles):
        row, col = i // 2, i % 2
        axs_metrics[row, col].set_title(title)

    # Track if we have any data across all plots
    global_has_data = False

    # Process each algorithm and watermark type
    for alg_idx, algorithm in enumerate(algorithms):
        for wm_idx, watermark_type in enumerate(watermark_types):
            # Dictionary to store channel data for plotting
            channel_data = {channel: {'bits': [], 'psnr': [], 'ssim': []} for channel in channels}

            # Process each channel and bit combination
            for channel in channels:
                for bit in bits:
                    watermarked_image_path = os.path.join(
                        base_image_paths[f'{algorithm}_{watermark_type}_{channel.lower()}'],
                        f"{channel}_host_image_with_{'SUTECH' if watermark_type == 'sutech' else 'RANDOM'}_watermark_bit_{bit}.tiff"
                    )

                    # Check if file exists before attempting to load
                    if os.path.exists(watermarked_image_path):
                        # Load images
                        watermarked_image = load_image(watermarked_image_path)

                        # Calculate PSNR between watermarked and original host
                        mse = np.mean((watermarked_image.astype(float) - host_image.astype(float)) ** 2)
                        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

                        # Calculate SSIM between watermarked and original host
                        # For multi-channel images, calculate SSIM for the specific channel
                        if len(host_image.shape) > 2 and host_image.shape[2] >= 3:
                            # Get channel index
                            channel_idx = {'RED': 0, 'GREEN': 1, 'BLUE': 2}.get(channel, 0)

                            # Extract the specific channel
                            host_channel = host_image[:, :, channel_idx]
                            watermarked_channel = watermarked_image[:, :, channel_idx]

                            # Calculate SSIM for this channel
                            ssim_value = ssim(host_channel, watermarked_channel, data_range=255)
                        else:
                            # For grayscale images or other formats
                            ssim_value = ssim(host_image, watermarked_image, data_range=255)

                        # Store the data for plotting
                        channel_data[channel]['bits'].append(bit)
                        channel_data[channel]['psnr'].append(psnr)
                        channel_data[channel]['ssim'].append(ssim_value)

            # Get axis for current subplot
            ax = axs_metrics[alg_idx, wm_idx]

            # Create a second y-axis for SSIM
            ax2 = ax.twinx()

            # Plot the results for each channel
            has_data = False
            for channel in channels:
                bits_list = channel_data[channel]['bits']
                psnr_values = channel_data[channel]['psnr']
                ssim_values = channel_data[channel]['ssim']

                # Set color based on channel
                if len(bits_list) > 0:
                    has_data = True
                    global_has_data = True

                    # Match color to channel name
                    if channel == 'RED':
                        color = 'red'
                    elif channel == 'GREEN':
                        color = 'green'
                    elif channel == 'BLUE':
                        color = 'blue'
                    else:
                        color = None  # Default matplotlib color

                    # Plot PSNR on left y-axis
                    ax.plot(bits_list, psnr_values, marker='o', linestyle='-',
                                     color=color, label=f"{channel} (PSNR)", linewidth=2)

                    # Plot SSIM on right y-axis with dashed line
                    ax2.plot(bits_list, ssim_values, marker='s', linestyle='--',
                                      color=color, label=f"{channel} (SSIM)", linewidth=2)

            # Set axis labels and formatting
            ax.set_xlabel('Bit Plane')
            ax.set_ylabel('PSNR (dB)')
            ax2.set_ylabel('SSIM')

            # Set y-limits
            if has_data:
                ax.set_ylim(bottom=20)  # PSNR values less than 20dB typically indicate poor quality
                ax2.set_ylim(0, 1.05)  # SSIM ranges from 0 to 1

                # Create combined legend for both axes
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Set x-ticks to only show the bit values we're using
            ax.set_xticks(bits)

    # Add a main title
    fig_metrics.suptitle('Host Image Quality Analysis (PSNR & SSIM)', fontsize=16, y=0.98)

    # Adjust layout without using tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    # Save figure
    output_path = os.path.join(base_image_paths['evaluation'], 'host_image_quality_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if global_has_data:
        print(f"{50 * '-'}\n\033[91mHost image quality evaluation saved to {output_path}\033[0m")
    else:
        print(f"{50 * '-'}\n\033[91mNo data available for quality evaluation. Check paths and file names.\033[0m")

    create_visual_comparison(base_image_paths, host_image, algorithms, watermark_types, channels)
    return


def create_visual_comparison(base_image_paths, host_image, algorithms, watermark_types, channels):
    """
    Create visual comparison between host image and clean images for selected bit planes.

    Args:
        base_image_paths: Dictionary of directory paths
        host_image: Original host image
        algorithms: List of algorithms used
        watermark_types: List of watermark types
        channels: List of color channels
    """
    # Choose representative bit planes (e.g., 0 for LSB, 4 for middle, 7 for MSB)
    representative_bits = range(8)

    # For each algorithm and watermark type, create a visual comparison
    for algorithm in algorithms:
        for watermark_type in watermark_types:
            for channel in channels:
                # Create a figure for this combination
                fig, axs = plt.subplots(2, len(representative_bits) + 1, figsize=(20, 8))

                # Set the title
                fig.suptitle(f'{algorithm.title().replace("_"," ")} Location - {watermark_type.upper()} Watermark - {channel} Channel',
                             fontsize=16, y=0.98)

                # Plot original host image in the first column
                axs[0, 0].imshow(host_image.astype(np.uint8))
                axs[0, 0].set_title('Original Host Image', fontsize=12)
                axs[0, 0].axis('off')

                # Calculate difference map for host image (will be all zeros)
                diff_host = np.zeros_like(host_image.astype(np.uint8))
                axs[1, 0].imshow(diff_host, cmap='viridis')
                axs[1, 0].set_title('Reference', fontsize=12)
                axs[1, 0].axis('off')

                # For each representative bit, get the clean image and plot
                for i, bit in enumerate(representative_bits):
                    clean_image_path = os.path.join(
                        base_image_paths[f'{algorithm}_{watermark_type}'],
                        'Extracted',
                        channel,
                        f'extracted_{"SUTECH" if watermark_type == "sutech" else "RANDOM"}_clean_image_bit_{bit}.tiff'
                    )

                    if os.path.exists(clean_image_path):
                        # Load clean image
                        clean_image = load_image(clean_image_path)

                        # Plot clean image
                        axs[0, i + 1].imshow(clean_image)
                        axs[0, i + 1].set_title(f'Clean Image (Bit {bit})', fontsize=12)
                        axs[0, i + 1].axis('off')

                        # Calculate and plot difference map
                        diff_image = np.abs(host_image.astype(float) - clean_image.astype(float))
                        # Normalize for better visualization
                        if np.max(diff_image) > 0:
                            diff_image = diff_image / np.max(diff_image) * 255

                        diff_image = diff_image.astype(np.uint8)
                        axs[1, i + 1].imshow(diff_image, cmap='viridis')
                        axs[1, i + 1].set_title(f'Difference Map (Bit {bit})', fontsize=12)
                        axs[1, i + 1].axis('off')
                    else:
                        # If image doesn't exist, show placeholders
                        axs[0, i + 1].text(0.5, 0.5, 'Image not available',
                                           ha='center', va='center', transform=axs[0, i + 1].transAxes)
                        axs[0, i + 1].set_title(f'Clean Image (Bit {bit})')
                        axs[0, i + 1].axis('off')

                        axs[1, i + 1].text(0.5, 0.5, 'Difference not available',
                                           ha='center', va='center', transform=axs[1, i + 1].transAxes)
                        axs[1, i + 1].set_title(f'Difference Map (Bit {bit})')
                        axs[1, i + 1].axis('off')

                # Adjust spacing
                plt.subplots_adjust(wspace=0.1, hspace=0.2)

                # Save the figure
                output_path = os.path.join(
                    base_image_paths['evaluation'],
                    f'visual_comparison_{algorithm}_{watermark_type}_{channel}.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Visual comparison saved to {output_path}")


def main() -> None:
    """Main function to execute the watermarking process."""
    print(50 * "-", "\n\033[91mInitializing Digital Watermarking System...\033[0m")

    # Create directory structure
    base_image_paths = create_directory_structure()

    # Process host image
    host_image = process_host_image(base_image_paths)

    # Process and embed watermarks
    binary_watermark, pseudo_random_image, fixed_locations, random_locations_sutech, random_locations_random = (
        process_watermarks(base_image_paths, host_image)
    )

    # Extract watermarks
    extract_watermarks(
        base_image_paths,
        host_image,
        binary_watermark,
        pseudo_random_image,
        fixed_locations,
        random_locations_sutech,
        random_locations_random
    )

    # Evaluate watermarks
    evaluate_watermarks(base_image_paths, binary_watermark, pseudo_random_image)
    evaluate_host_image(base_image_paths, host_image)
    print(50 * "-", "\n\033[91mDigital Watermarking Process Completed Successfully!\033[0m")


if __name__ == "__main__":
    main()