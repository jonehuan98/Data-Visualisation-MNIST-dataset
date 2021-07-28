#License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)






import os
import struct
import numpy as np
 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels


# unzips mnist

import sys
import gzip
import shutil

if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'


zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
print(zipped_mnist)
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 




X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))






import matplotlib.pyplot as plt

#PLEASE WRITE YOUR CODE HERE...

# define subplots of 10 rows and 5 columns
fig, ax = plt.subplots(nrows=10, ncols=5, sharex=True, sharey=True)

# ax is in the form of 10 nested lists with 5 AxesSubPlot elements

for i in range(10): # for each number 1-9
    for j in range(5): # for each index 0-4
        img = X_train[y_train == i][j].reshape(28, 28) #get train image from label index
        ax[i,j].imshow(img, cmap='Greys') # assign according to subplot

ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
plt.tight_layout()
plt.show()
