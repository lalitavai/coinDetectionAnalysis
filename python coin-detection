import cv2
import matplotlib.pyplot as plt

import numpy as np


# Image path
imagePath = "CoinsB.png"

image=cv2.imread(imagePath)

# Convert to grayscale
# Store in variable imageGray
###
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
###
#cv2.imshow("imageGray", imageGray)
# Split the image into R, G, and B channels
imageB, imageG, imageR = cv2.split(image)


thred, imRed = cv2.threshold(imageR, 175 ,255, cv2.THRESH_BINARY)
thgreen, imGreen = cv2.threshold(imageG, 90 ,255, cv2.THRESH_BINARY)
thblue, imBlue = cv2.threshold(imageB, 124 ,255, cv2.THRESH_BINARY)

#cv2.imshow("imRth", imRth)
#cv2.imshow("imGth", imGth)
#cv2.imshow("imBlue", imBlue)





# get kernels :morphological operations

ellipseKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
###


# Remove white holes
#imcopy = maxImg.copy()
imcopy = imBlue.copy()

morphOpenImg = cv2.morphologyEx(imcopy, cv2.MORPH_OPEN, rectKernel, iterations = 5)

# cv2.imshow("o_5x5", o_5x5)

# remove black holes
# remove black holes
morphImgCopy = morphOpenImg.copy()
morphCloseImg = cv2.morphologyEx(morphImgCopy, cv2.MORPH_CLOSE, rectKernel, iterations = 5)
plt.subplot(111)
plt.imshow(morphCloseImg)

# cv2.imshow("c_5x5", c_5x5)



# remove coin centers
morphOpenClose = morphCloseImg.copy()

morphOpenClose = cv2.morphologyEx(morphOpenClose, cv2.MORPH_OPEN, rectKernel, iterations = 25)

plt.subplot(111)
plt.imshow(morphOpenClose)


morphOpenClose = cv2.erode(morphOpenClose, rectKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("erode image")


morphOpenClose = cv2.dilate(morphOpenClose, ellipseKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("dilate image")


morphOpenClose = cv2.erode(morphOpenClose, rectKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("erode image again")


morphOpenClose = cv2.dilate(morphOpenClose, ellipseKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("dilate image aagin")


morphOpenClose = cv2.erode(morphOpenClose, rectKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("erode image again")


morphOpenClose = cv2.dilate(morphOpenClose, ellipseKernel, iterations = 32)
plt.subplot(111)
plt.imshow(morphOpenClose)
plt.title("dilate image again")



#revert image
morphImgCopyRevertImg = morphOpenClose.copy()
morphImgCopyRevertImg = cv2.bitwise_not(morphImgCopyRevertImg)

plt.subplot(111)
plt.imshow(morphImgCopyRevertImg)
plt.title("revert image ")

#cv2.imshow("c_5x5_2_not", c_5x5_2_not)


# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

###
keypoints = detector.detect(morphOpenClose)
###

print('There are {} number of coins detected'.format(len(keypoints)))


###
imageWithCircles = image.copy()

for k in keypoints:
    x, y = k.pt
    x = int(round(x))
    y = int(round(y))
    # Mark center in BLACK
    cv2.circle(imageWithCircles, (x, y), 25, (255, 0,0 ), -1)
    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter / 2))
    # Mark blob in RED
    cv2.circle(imageWithCircles, (x, y), radius, (0, 0, 255), 15)

###
# Resize the image (e.g., to 50% of its original size)
width1 = int(imageWithCircles.shape[1] * 0.15)  # 50% of the original width
height1 = int(imageWithCircles.shape[0] * 0.15) # 50% of the original height
dim1 = (width1, height1)

# Resize the image
resized_imgWithCircles = cv2.resize(imageWithCircles, dim1, interpolation=cv2.INTER_AREA)

# Display the resized image
cv2.imshow("Image With Circles", resized_imgWithCircles)


_, coinlabels = cv2.connectedComponents(morphImgCopyRevertImg)
print('Number of connected components detected = {}'.format(coinlabels.max()))
###

def displayConnectedComponents(im):
    """
    Displays connected components in an image with applied colormap for better
    visual differentiation of components. It normalizes the connected component
    labels, scales them for visualization, converts the image to 8-bit format,
    and applies a color map.

    :param im: The input image containing connected component labels.
               It is expected to be a single-channel image with integer values.
    :return: None
    """
    imLabels = im
    # The following line finds the min and max pixel values
    # and their locations in an image.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)
    # Normalize the image so the min value is 0 and max value is 255.
    imLabels = 255 * (imLabels - minVal) / (maxVal - minVal)
    # Convert image to 8-bits unsigned type
    imLabels = np.uint8(imLabels)
    # Apply a color map
    imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
    # Display colormapped labels
    plt.imshow(imColorMap[:, :, ::-1])


# Display connected components using displayConnectedComponents
displayConnectedComponents(coinlabels)

contours, hierarchy = cv2.findContours(morphOpenClose, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )


print('Number of contours found = ' + str(len(contours)))

# Draw all contours
img = image.copy()
cv2.drawContours(img, contours, -1, (0,255,0), 25)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Contours found image")

for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))

    area = cv2.contourArea(contours[-1])
    print('Maximum area of contour = ' + str(area))


img = image.copy()

filtered_contours = []
for index, cnt in enumerate(contours):
    a = cv2.contourArea(cnt)
    if  area != a:
        filtered_contours.append(cnt)

cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 15)
# cv2.imshow("img2", img)


img = image.copy()
cv2.drawContours(img, filtered_contours, -1, (0,255,0), 15)

#cv2.imshow("Image 1", img)

# print sorted area
area_lst = []
for cnt in filtered_contours:
    a = cv2.contourArea(cnt)
    area_lst.append(a)
sorted(area_lst)
for a in area_lst:
    print(a)
    
img = image.copy()
cv2.drawContours(img, filtered_contours, -1, (0,255,0), 15)
index = 0
for k in keypoints:
    index += 1
    x, y = k.pt
    size = k.size
    cv2.circle(img, (int(x),int(y)), 15, (0,0,255), -1)
    cv2.putText(img, "{}".format(index), ((int(x+40)), (int(y-10))), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255),5)


# Resize the image (e.g., to 50% of its original size)
width = int(img.shape[1] * 0.15)  # 50% of the original width
height = int(img.shape[0] * 0.15) # 50% of the original height
dim = (width, height)

# Resize the image
resized_img2 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Display the resized image
cv2.imshow("Image with Contours", resized_img2)




c = cv2.waitKey(0)
cv2.destroyAllWindows()
