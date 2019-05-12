import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import skimage.data
import matplotlib.pyplot as plt

def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """

    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im_mask)  # extract neighbouring information
    plt.show()

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax1.imshow(im_orig)  # extract neighbouring information
    plt.show()

    return im_orig


def selective_search(im_orig, scale, sigma, min_size):
    img = _generate_segments(im_orig, scale, sigma, min_size)
    return

if __name__ == '__main__':
    img = skimage.data.astronaut()
    selective_search(img, scale=600, sigma=0.9, min_size=6)