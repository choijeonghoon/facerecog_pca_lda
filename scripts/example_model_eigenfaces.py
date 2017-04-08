import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.util import read_images
from tinyfacerec.model import EigenfacesModel

# read images
[X,y] = read_images('training')
[A,b] = read_images('test')

# compute the eigenfaces model
model = EigenfacesModel(X[:], y[:])
# get a prediction for the first observation
c=[]
num_correct=0
for i in range(120):
    if i%3==0:
        a=int(i/3)
        print ("expected =", y[(a)*7])
    print("/", "predicted =", model.predict(A[i]))
    c.append(model.predict(A[i]))
    
    if c[i]==b[i]:
        num_correct+=1


num_test = float(len(b[:]))
accuracy = float(num_correct)/ num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))