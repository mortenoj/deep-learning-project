# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv.imread('example.png',0)
# img2 = img.copy()

# template = cv.imread('ak2.png',0)
# w, h = template.shape[::-1]

# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            # 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# for meth in methods:
    # img = img2.copy()
    # method = eval(meth)
    # # Apply template Matching
    # res = cv.matchTemplate(img,template,method)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        # top_left = min_loc
    # else:
        # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(img,top_left, bottom_right, 255, 2)

    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    # plt.suptitle(meth)
    # plt.show()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageChops



def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


# img = Image.open('ak.png')
# wpercent = 1.25

# img = Image.open('awp.png')
# wpercent = 1.77

img = Image.open('krieg.png')
wpercent = 1.2

# wpercent = 0.85

img = trim(img)

hsize = int((float(img.size[1])*float(wpercent)))
wsize = int((float(img.size[0])*float(wpercent)))
img = img.resize((wsize, hsize), Image.ANTIALIAS)
img.save('res.png')

img_rgb = cv.imread('./game4.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('res.png',0)


w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

threshold = 0.72
loc = np.where( res >= threshold)
i = 0
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    i += 1
cv.imwrite('res.png',img_rgb)
print(i)
