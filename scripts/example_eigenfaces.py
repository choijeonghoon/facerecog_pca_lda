import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.subspace import pca
from tinyfacerec.util import normalize, asRowMatrix, read_images
from tinyfacerec.visual import subplot

'''
if __name__ == '__main__':

   if len(sys.argv) != 2:
      print ("USAGE: example_eigenfaces.py </path/to/images>")
      sys.exit()
'''    
    # read images
[X,y] = read_images('att_faces')

# perform a full pca
[D, W, mu] = pca(asRowMatrix(X), y)

import matplotlib.cm as cm

# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)
E = []
my_cnt = 0
for i in range(min(len(X), 400)):
    e = W[:,i].reshape(X[0].shape)
    
    if i % 10 == 0 and i != 0:
        subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=3, cols=4, sptitle="Eigenface", colormap=cm.gray, filename="python_pca_eigenfaces" + str(my_cnt) + ".png")
        E = []
        my_cnt += 1
    E.append(normalize(e,0,255))
    # plot them and store the plot to "python_eigenfaces.pdf"    
    

from tinyfacerec.subspace import project, reconstruct

# reconstruction steps
steps=[i for i in range(10, min(len(X), 320), 20)]
E = []
for i in range(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(W[:,0:numEvs], X[0].reshape(1,-1), mu)
    R = reconstruct(W[:,0:numEvs], P, mu)
    # reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append(normalize(R,0,255))
# plot them and store the plot to "python_reconstruction.pdf"
subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="python_pca_reconstruction.png")