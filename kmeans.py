from PIL import Image, ImageOps
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
# np.set_printoptions(threshold=sys.maxsize)
import csv

# ############# PREPROCESSING ################################
# images are 64x64 to start but only use the bottom right corner

x_data = []
for i in range(18395):
    num = "{:04d}".format(i)
    image = Image.open('data1/NIT_MPS_1_' + num + '.bmp')
    im = image.crop((14, 24, 64, 64))  # crop to get the actual image
    im = ImageOps.flip(im)
    pix = np.array(im) / 255.0
    x_data.append(pix)

x_train = np.array(x_data).reshape(len(x_data), -1)
print('X_train shape: ', x_train.shape)
# ############ K MEANS #########################################

CLUSTERS = 7

kmeans = KMeans(n_clusters=CLUSTERS)

kmeans.fit(x_train)

lab = kmeans.labels_

total_im = 0
cats = []
tots = []
indexes = []

for i in range(CLUSTERS):
    index = np.where(lab == i)
    print('catagory : ', i)
    print('Number of images in catagory : ', len(lab[index]))
    print('image number : ', index)
    print('middle index : ', index[0][int(len(index[0])/2)])
    total_im += len(lab[index])
    indexes.append(index)
    tots.append(len(lab[index]))
    cats.append(str(i))
    #print(index[0][0])
    num = "{:04d}".format(index[0][0])
    pic = Image.open('data1/NIT_MPS_1_' + num + '.bmp')
    pic.show()

print(total_im)

plt.bar(cats, tots)
plt.title('Image Cluster Distribution')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Images')
plt.show()

# num = "{:04d}".format(629)
# pic = Image.open('data2/NIT_MPS_2_' + num + '.bmp')
# pic.show()
