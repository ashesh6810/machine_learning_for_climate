import numpy as np
from numpy import genfromtxt
x=genfromtxt('training_40ensemble_4classes_fullZ.csv',delimiter=',')
np.save('training_40ensemble_4classes',x)
y=genfromtxt('labels_40ensemble_4classes_fullZ.csv',delimiter=',')
np.save('labels_40ensemble_4classes',y)
xx=genfromtxt('test_40ensemble_4classes_fullZ.csv',delimiter=',')
np.save('test_40ensemble_4classes',xx)
yy=genfromtxt('test_labels_40ensemble_4classes_fullZ.csv',delimiter=',')
np.save('test_labels_40ensemble_4classes',yy)
print('done')
