import numpy as np
from numpy import genfromtxt

test_y=np.load('test_labels_40ensemble_4classes.npy')
#test_y=test_label[range(int(3*(np.size(test_label,0))/4),np.size(test_label,0)),:]
prediction=genfromtxt('prediction_TRUE_4layers.csv', delimiter=',')

#for i in range(0,np.size(prediction,0)):
# for j in range(0,np.size(prediction,1)):
#  if(prediction[i,j]<=0.5):
#   prediction[i,j]=0
#  if(prediction[i,j]>0.5):
#   prediction[i,j]=1


for i in range(0,np.size(prediction,0)):
 x=np.argmax(prediction[i,:])
 prediction[i,x]=1
 
for i in range(0,np.size(prediction,0)):
 for j in range(0,np.size(prediction,1)):
   if (prediction[i,j]<>1):
    prediction[i,j]=0

  
class1=0
class2=0
class3=0
class4=0  

class1_true=0
class2_true=0
class3_true=0
class4_true=0

for i in range(0,np.size(test_y,0)):
    if (test_y[i,0]==1):
        class1=class1+1
    if (test_y[i,1]==1):
        class2=class2+1
    if (test_y[i,2]==1):
        class3=class3+1
    if (test_y[i,3]==1):
        class4=class4+1

print ('class1=')
print (class1)

print ('class2=')
print (class2)

print ('class3=')
print (class3)

print ('class4=')
print (class4)





for i in range(0,np.size(test_y,0)):
 if (test_y[i,0]): 
  if(prediction[i,0]==1):
   class1_true=class1_true+1
 if (test_y[i,1]):
  if(prediction[i,1]==1):
   class2_true=class2_true+1
 if (test_y[i,2]):
  if(prediction[i,2]==1):
   class3_true=class3_true+1 
 if (test_y[i,3]):
  if(prediction[i,3]==1):
   class4_true=class4_true+1

print('class1_true=')
print(class1_true)

print('class2_true=')
print(class2_true)

print('class3_true=')
print(class3_true)

print('class4_true=')
print(class4_true)
