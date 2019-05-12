import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Datas.SelectiveSearch as selectivesearch
import skimage.data
import numpy as np
from PIL import Image as pilimage

def main():

    # loading astronaut image
    im = pilimage.open(r'C:\Users\BULUT\Desktop\IMG-20190313-WA0000.jpg')
    im = im.resize((500, 500), pilimage.ANTIALIAS)
    im = np.asarray(im, dtype='uint8')

    img_lbl, regions = selectivesearch.selective_search(
        im, scale=300, sigma=2, min_size=100)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    delete_supererogator_rect(candidates)
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

def delete_supererogator_rect(candidates):
    deleted = set()
    for rectx in candidates:
        x, y, w, h = rectx
        for rect in candidates:
            if rect != rectx:
                x0, y0, w0, h0 = rect
                if x0 >= x and (x0 + w0) <= (x + w) and y0 >= y and (y0 + h0) <= (y + h):
                    if rectx not in deleted:
                        deleted.add(rectx)


    for i in deleted:
        candidates.remove(i)

if __name__ == "__main__":
    main()
