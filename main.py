import os.path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageChannels:
    RED: int = 0
    GREEN: int = 1
    BLUE: int = 2


def generate_pseudo_random_image(
        shape: tuple,
        is_binary: bool
) -> np.ndarray:

    if len(shape) != 2:
        raise ValueError('Shape must be 2d')

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

    return Image.fromarray(image_array).save(filename)


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
        raise ValueError('Shape must be 3d')

    binary_array = np.zeros(image_array[:, :, channel_index].shape + (8,))
    for bit in range(8):
        binary_array[:, :, bit] = np.bitwise_and(image_array[:, :, channel_index], 2**bit) > 0

    return np.array(binary_array, dtype=np.bool)


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
    plt.show()
    return


def main():
    base_image_path = "Images"
    if not os.path.isdir(base_image_path):
        os.mkdir(base_image_path)

    # Creating a pseudo random image
    pseudo_random_image = generate_pseudo_random_image(
        shape=(64, 64),
        is_binary=True
    )
    save_image(
        pseudo_random_image,
        filename=os.path.join(base_image_path,'pseudo_random_image.png')
    )

    # Creating the host image, resize and so on.
    host_image = load_image(
        image_path=os.path.join(
            base_image_path,
            'host_image.jpeg'
        )
    )
    resized_host_image = resize_image(
        image_array=host_image,
        target_dimensions=(512, 512)
    )
    save_image(
        resized_host_image,
        filename=os.path.join(base_image_path,'resized_host_image.png')
    )

    # plot LSB to MSB of each channel(r, g, b) in resized host data
    channel_configs = [
        (ImageChannels.RED, 'red_LSB_to_MSB_images.png'),
        (ImageChannels.GREEN, 'green_LSB_to_MSB_images.png'),
        (ImageChannels.BLUE, 'blue_LSB_to_MSB_images.png')
    ]

    for channel, output_filename in channel_configs:
        binary_images = extract_binary_images(
            image_array=resized_host_image,
            channel_index=channel
        )
        plot_bit_planes(
            binary_array=binary_images,
            filename=os.path.join(base_image_path, output_filename)
        )


if __name__ == "__main__":
    main()