import cleanup
import padding
import deband_full_batch


if __name__ == '__main__':
    image_sizes = {}

    cleanup.cleanup()
    cleanup.setup('f')
    padding.pad_images(image_sizes, 'f')
    print("batch debanding")
    deband_full_batch.deband_images_batch(image_sizes, 0)
    cleanup.cleanup()