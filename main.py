import os.path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageChannels:
    RED: int = 0
    GREEN: int = 1
    BLUE: int = 2

class WatermarkedImageGroup:
    SUTECH: int = 0
    RANDOM: int = 1


def generate_pseudo_random_image(
        shape: tuple,
        is_binary: bool
) -> np.ndarray:

    if len(shape) != 2:
        raise ValueError('Image array must be two-dimensional')

    random_image = np.array(
            127.5 * np.random.randn(*shape) + 127.5,
            dtype=np.uint8
        )
    return random_image > 127.5 if is_binary else random_image


def save_image(
        image_array: np.ndarray,
        filename: str
) -> None:

    if len(image_array.shape) not in (2, 3):
        raise ValueError('Image array must be two or three-dimensional')
    Image.fromarray(image_array).save(filename)
    return print(50*"-", "\nImage saved as {}".format(filename))


def load_image(
        image_path: str
) -> np.ndarray:

    if not os.path.isfile(image_path):
        raise ValueError('Specified image file does not exist')

    return np.array(Image.open(image_path))


def resize_image(
        image_array: np.ndarray,
        target_dimensions: tuple[int, int]
) -> np.ndarray:

    if len(image_array.shape) not in (2, 3):
        raise ValueError('Image array must be two or three-dimensional')

    return np.array(Image.fromarray(image_array).resize(target_dimensions))


def extract_binary_images(
        image_array: np.ndarray,
        channel_index: int
) -> np.ndarray:

    if len(image_array.shape) != 3:
        raise ValueError('Image array must be three-dimensional')

    binary_array = np.zeros(image_array[:, :, channel_index].shape + (8,))
    for bit in range(8):
        binary_array[:, :, bit] = np.bitwise_and(image_array[:, :, channel_index], 2**bit) > 0

    return np.array(binary_array, dtype=np.bool)


def generate_binary_image_from_gray_scale(
        image_array: np.ndarray,
) -> np.ndarray:
    return np.where(image_array[:,:,0] > 127, 255, 0).astype(np.bool)


def plot_bit_planes(
        binary_array: np.ndarray,
        filename: Optional[str] = None,
) -> None:

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
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        return print(50*"-","\nImage saved as {}".format(filename))
    plt.close()
    return

def embed_watermark(
    host_image: np.ndarray,
    watermark: np.ndarray,
    pixel_locations: List[Tuple[int, int]],
    bit_plane: int,
    channel: int
) -> np.ndarray:
    # Validate inputs
    assert 0 <= bit_plane <= 7, f"Invalid bit plane: {bit_plane}"
    assert 0 <= channel < host_image.shape[2], f"Invalid channel: {channel}"
    print(len(pixel_locations), watermark.size)
    assert len(pixel_locations) == watermark.size, "Pixel locations must match watermark size"

    host_copy = host_image.copy()
    selected_channel = host_copy[:, :, channel]

    # Create the bit mask to clear the target bit
    clear_mask = ~(1 << bit_plane) & 0xFF

    print(f"Embedding watermark of shape {watermark.shape}")
    print(f"Using bit plane {bit_plane} on channel {channel}")

    embedding_errors = 0
    for idx, (i, j) in enumerate(pixel_locations):
        if i < host_image.shape[0] and j < host_image.shape[1]:
            pixel_cleared = selected_channel[i, j] & clear_mask  # Clear bit
            watermark_bit = 1 if watermark.flat[idx] > 0 else 0
            selected_channel[i, j] = pixel_cleared | (watermark_bit << bit_plane)
        else:
            embedding_errors += 1
            print(f"Warning: Pixel location {(i,j)} out of bounds")

    if embedding_errors > 0:
        print(f"Failed to embed {embedding_errors} pixels")

    host_copy[:, :, channel] = selected_channel
    return host_copy

def extract_watermark(
    watermarked_image: np.ndarray,
    pixel_locations: List[Tuple[int, int]],
    bit_plane: int,
    channel: int,
    watermark_shape: Tuple[int, int]
) -> np.ndarray:

    extracted_watermark = np.zeros(watermark_shape, dtype=np.bool)

    selected_channel = watermarked_image[:, :, channel]

    for idx, (i, j) in enumerate(pixel_locations):
        extracted_watermark.flat[idx] = (selected_channel[i, j] >> bit_plane) & 1 # Extract bit

    return extracted_watermark


def plot_watermarked_images(
        base_image_path: str,
        image_group: int,
        filename: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    image_channel_path = base_image_path.split(r"/")[-1]
    for bit in range(8):
        row, col = divmod(bit, 4)  # Arrange in 2 rows, 4 columns

        if image_group == 0:
            # Load the image with SUTECH watermark
            image_channel = image_channel_path.split("SUTECH")[-1]
            image_path = base_image_path + image_channel + f'_host_image_with_SUTECH_watermark_bit_{bit}.tiff'
            image = np.array(Image.open(image_path))
        else:
            # Load the image with RANDOM watermark
            image_channel = image_channel_path.split("Random")[-1]
            image_path = base_image_path + image_channel + f'_host_image_with_RANDOM_watermark_bit_{bit}.tiff'
            image = np.array(Image.open(image_path))
        axes[row, col].imshow(image)  # Display in grayscale
        axes[row, col].set_title(f'Bit {bit} - SUTECH')
        axes[row, col].axis('off')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)

    plt.close()

def main():
    base_image_paths = {
        "root": r"Images",
        "host": r"Images\Host Image",
        "bit_planes": r"Images\Host Image\Bit Planes",
        "watermarked": r"Images\Watermarked Images",
        "random": r"Images\Watermarked Images\Random",
        "random_red": r"Images\Watermarked Images\Random\RED",
        "random_green": r"Images\Watermarked Images\Random\GREEN",
        "random_blue": r"Images\Watermarked Images\Random\BLUE",
        "sutech": r"Images\Watermarked Images\SUTECH",
        "sutech_red": r"Images\Watermarked Images\SUTECH\RED",
        "sutech_green": r"Images\Watermarked Images\SUTECH\GREEN",
        "sutech_blue": r"Images\Watermarked Images\SUTECH\BLUE",
    }

    # Ensure all directories exist
    for path in base_image_paths.values():
        os.makedirs(path, exist_ok=True)

    # Creating a pseudo random image
    pseudo_random_image = generate_pseudo_random_image(
        shape=(64, 64),
        is_binary=True
    )
    save_image(
        pseudo_random_image,
        filename=os.path.join(base_image_paths['root'],'pseudo_random_image.tiff')
    )

    # Creating the host image, resize and so on.
    host_image = load_image(
        image_path=os.path.join(
            base_image_paths['root'],
            'host_image.jpeg'
        )
    )
    resized_host_image = resize_image(
        image_array=host_image,
        target_dimensions=(512, 512)
    )
    save_image(
        resized_host_image,
        filename=os.path.join(base_image_paths['host'],'resized_host_image.tiff')
    )

    # plot LSB to MSB of each channel(r, g, b) in resized host data
    channel_configs = [
        (ImageChannels.RED, 'red_gray_scale.tiff' ,'red_LSB_to_MSB_images.tiff'),
        (ImageChannels.GREEN, 'green_gray_scale.tiff' ,'green_LSB_to_MSB_images.tiff'),
        (ImageChannels.BLUE, 'blue_gray_scale.tiff' ,'blue_LSB_to_MSB_images.tiff')
    ]

    for channel, output_grayscale_filename, output_binary_filename in channel_configs:
        save_image(
            resized_host_image[:,:,channel],
            filename=os.path.join(base_image_paths['host'], output_grayscale_filename)
        )
        binary_images = extract_binary_images(
            image_array=resized_host_image,
            channel_index=channel
        )
        plot_bit_planes(
            binary_array=binary_images,
            filename=os.path.join(base_image_paths['bit_planes'], output_binary_filename)
        )


    # Load Watermark
    watermark = load_image(
        image_path=os.path.join(base_image_paths['root'], 'sutech.jpg'),
    )
    print(50*"-","\nWatermark shape:",watermark.shape)
    binary_watermark = generate_binary_image_from_gray_scale(
        image_array=watermark
    )
    save_image(
        image_array=binary_watermark,
        filename=os.path.join(base_image_paths['root'], 'binary_sutech_watermark.tiff')
    )
    print(50 * "-", "\nBinary watermark:\n", binary_watermark)

    channels = {
        ImageChannels.RED: {
        'base_address': "red",
        'save_address': "RED_host_image_with_"
    },
        ImageChannels.GREEN: {
        'base_address': "green",
        'save_address': "GREEN_host_image_with_"
    },
        ImageChannels.BLUE: {
        'base_address': "blue",
        'save_address': "BLUE_host_image_with_"
        }
    }

    # Define fixed locations (a 64x64 block)
    pixel_locations = [(i, j) for i in range(binary_watermark.shape[0]) for j in range(binary_watermark.shape[1])]
    for bit in range(8):
        for channel, address in channels.items():

            print(binary_watermark.shape)
            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=binary_watermark,
                pixel_locations=pixel_locations,
                bit_plane=bit,
                channel=channel,
            )
            print(50 * '-', f"\nHost image with SUTECH watermark embedded (bit plane {bit}):\n", watermarked_image)
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['sutech_' + address['base_address']],
                    address["save_address"] + f'SUTECH_watermark_bit_{bit}.tiff'
                )
            )

            watermarked_image = embed_watermark(
                host_image=host_image,
                watermark=pseudo_random_image,
                pixel_locations=pixel_locations,
                bit_plane=bit,
                channel=channel
            )
            print(50 * '-', f"\nHost image with pseudo random watermark embedded (bit plane {bit}):\n", watermarked_image)
            save_image(
                image_array=watermarked_image,
                filename=os.path.join(
                    base_image_paths['random_' + address['base_address']],
                    address["save_address"] + f'RANDOM_watermark_bit_{bit}.tiff'
                )
            )

    for address in channels.values():
        plot_watermarked_images(
            base_image_path=base_image_paths['sutech_' + address['base_address']],
            image_group=WatermarkedImageGroup.SUTECH,
            filename=os.path.join(base_image_paths['sutech'], address["save_address"] + "SUTECH_watermarks.tiff")
        )
        plot_watermarked_images(
            base_image_path=base_image_paths['random_' + address['base_address']],
            image_group=WatermarkedImageGroup.RANDOM,
            filename=os.path.join(base_image_paths['random'], address["save_address"] + "RANDOM_watermarks.tiff")
        )

    extracted_watermark = extract_watermark(
        watermarked_image= np.array(
            Image.open(
                "Images/Watermarked Images/SUTECH/BLUE/BLUE_host_image_with_SUTECH_watermark_bit_0.tiff"
            )
        ),
        pixel_locations=pixel_locations,
        bit_plane=0,
        channel=ImageChannels.BLUE,
        watermark_shape=(64, 64)
    )
    save_image(
        image_array=extracted_watermark,
        filename="Images/test.tiff"
    )


if __name__ == "__main__":
    main()