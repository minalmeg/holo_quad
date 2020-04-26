import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('dark_background')
img = cv.imread('earth.jpg')[:,:,::-1]

(h, w) = img.shape[:2
                    ]

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (int(h/5),int(w/4),int(h/1.5),int(w/1.5))
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.axis('off')



# calculate the center of the image
center = (w / 2, h / 2)
angle0 = 0
angle90 = 90
angle180 = 180
angle270 = 270
 
scale = 1.5
 
# Perform the counter clockwise rotation holding at the center
# 0 degrees
M = cv.getRotationMatrix2D(center, angle0, scale)
rotated0 = cv.warpAffine(img, M, (h, w))

# 90 degrees
M = cv.getRotationMatrix2D(center, angle90, scale)
rotated90 = cv.warpAffine(img, M, (h, w))
 
# 180 degrees
M = cv.getRotationMatrix2D(center, angle180, scale)
rotated180 = cv.warpAffine(img, M, (w, h))
 
# 270 degrees
M = cv.getRotationMatrix2D(center, angle270, scale)
rotated270 = cv.warpAffine(img, M, (h, w))
 

plt.subplot(221),plt.imshow(rotated0,'gray'),plt.axis("off")
plt.subplot(222),plt.imshow(rotated270,'gray'),plt.axis("off")
plt.subplot(223),plt.imshow(rotated90,'gray'),plt.axis("off")
plt.subplot(224),plt.imshow(rotated180,'gray'),plt.axis("off")

plt.tight_layout(rect=[0.2, 0.0, 0.8, 1])

plt.show()