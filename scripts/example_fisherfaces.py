import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.subspace import fisherfaces
from tinyfacerec.util import normalize, asRowMatrix, read_images
from tinyfacerec.visual import subplot

# read images
[X,y] = read_images('att_faces')
# perform a full lda
[D, W, mu] = fisherfaces(asRowMatrix(X[1:]), y)

import matplotlib.cm as cm

# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)
E = []
for i in range(min(W.shape[1], 16)):
    e = W[:,i].reshape(X[0].shape)
    E.append(normalize(e,0,255))
# plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
subplot(title="Fisherfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.gray, filename="python_fisherfaces_fisherfaces.png")

from tinyfacerec.subspace import project, reconstruct

E = []
for i in range(min(W.shape[1], 16)):
    e = W[:,i].reshape(-1,1)
    P = project(e, X[0].reshape(1,-1), mu)
    R = reconstruct(e, P, mu)
    # reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append(normalize(R,0,255))
# plot them and store the plot to "python_reconstruction.pdf"
subplot(title="Fisherfaces Reconstruction", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.gray, filename="python_fisherfaces_reconstruction.png")
