\

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy
import statistics 
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def convertSL(list):
    ls= []
    for x in range(len(list)):
        ls.append(list[x][0])
    return ls

def convertSW(list):
    ls= []
    for x in range(len(list)):
        ls.append(list[x][1])
    return ls
def convertPL(list):
    ls= []
    for x in range(len(list)):
        ls.append(list[x][2])
    return ls
def convertPW(list):
    ls= []
    for x in range(len(list)):
        ls.append(list[x][3])
    return ls

def lk(list, mean, stdev):
    ls=[]
    prod=1
    for x in range(len(list)):
        for y in range(4):
           prod = prod * norm.pdf(list[x][y],mean[y],stdev[y])
        ls.append(prod)
        prod=1
            
    return ls

def Posterior(list, prior):
    ls=[]
    temp=[]
    
    for y in range(30):
            q =  (list[0][y]* prior)/((list[0][y]* prior) + (list[1][y]* prior) + (list[2][y]* prior))
            list[0][y]=q
    
    for y in range(30):
            q =  (list[1][y]* prior)/ ((list[0][y]* prior) + (list[1][y]* prior) + (list[2][y]* prior))
            list[1][y]=q
            
    for y in range(30):
            q =  (list[2][y]* prior)/ ((list[0][y]* prior) + (list[1][y]* prior) + (list[2][y]* prior))
            list[2][y]=q
            
    return list


def Evaluation(list):
    ls=[]
    for y in range(30):
        if(list[0][y]> list[1][y]  and list[0][y]> list[2][y]):
            ls.append(0)
        if(list[1][y]> list[0][y] and list[1][y]> list[2][y] ):
            ls.append(1)
        if (list[2][y]> list[0][y] and list[2][y]> list[1][y]):
            ls.append(2)
    return ls


# import some data to play with
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target


#print(iris)
#Cross Validation
X_train, X_test, y_train, y_test= train_test_split(x,y,test_size= 0.2, stratify =y)
Setosa=[]
Virginica =[]
Versicolor=[]
list = []
Liklihood=[]



#Liklihood
sepal_len= X_train.T[0]
sepal_wid= X_train.T[1]
petal_len= X_train.T[2]
petal_wid= X_train.T[3]


for x in range(120):
    if (y_train[x] == 0):
        Setosa.append(X_train[x])
 

for x in range(120):
    if (y_train[x] == 1):
        Virginica.append(X_train[x])


for x in range(120):
    if (y_train[x] == 2):
        Versicolor.append(X_train[x])

print("SETOSA FEATURES:")
meaan=sum(Setosa)/len(Setosa)

print("mean:",meaan)
list= convertSL(Setosa)
Septal_Length_Stdev = statistics.stdev(list)
print( "Septal Length Stdev:",Septal_Length_Stdev)
#print(list)        

list= convertSW(Setosa)
Septal_Width_Stdev = statistics.stdev(list)
print( "Septal width Stdev:",Septal_Width_Stdev)



list= convertPL(Setosa)
Petal_Length_Stdev = statistics.stdev(list)
print( "Petal length Stdev:",Petal_Length_Stdev)


list= convertPW(Setosa)
Petal_Width_Stdev = statistics.stdev(list)
print( "Petal width Stdev:",Petal_Width_Stdev)


Setosa_Stdev = [Septal_Length_Stdev,Septal_Width_Stdev,Petal_Length_Stdev, Petal_Width_Stdev]
Setosa_Mean = sum(Setosa)/len(Setosa)

v = lk(X_test,Setosa_Mean,Setosa_Stdev)
Liklihood.append(v)
print("")

#print(np.matrix(Liklihood))
#print("len of Liklihood: ",len(Liklihood))

print("")
print("Virginica FEATURES:")
print("mean : ",sum(Virginica)/len(Virginica))
list= convertSL(Virginica)
Septal_Length_Stdev = statistics.stdev(list)
print( "Septal Length Stdev:",Septal_Length_Stdev)       

list= convertSW(Virginica)
Septal_Width_Stdev = statistics.stdev(list)
print( "Septal width Stdev:",Septal_Width_Stdev)

list= convertPL(Virginica)
Petal_Length_Stdev = statistics.stdev(list)
print( "Petal length Stdev:",Petal_Length_Stdev)

list= convertPW(Virginica)
Petal_Width_Stdev = statistics.stdev(list)
print( "Petal width Stdev:",Petal_Width_Stdev)


Virginica_Stdev = [Septal_Length_Stdev,Septal_Width_Stdev,Petal_Length_Stdev, Petal_Width_Stdev]
Virginica_Mean = sum(Setosa)/len(Setosa)

v = lk(X_test,Virginica_Mean,Virginica_Stdev)
Liklihood.append(v)
print("")

#print(np.matrix(Liklihood))
#print("len of Liklihood: ",len(Liklihood))
#print("")


print("")
print("Versicolor FEATURES:")
print("mean : ",sum(Versicolor)/len(Versicolor))
list= convertSL(Versicolor)
Septal_Length_Stdev = statistics.stdev(list)
print( "Septal Length Stdev:",Septal_Length_Stdev)       

list= convertSW(Versicolor)
Septal_Width_Stdev = statistics.stdev(list)
print( "Septal width Stdev:",Septal_Width_Stdev)

list= convertPL(Versicolor)
Petal_Length_Stdev = statistics.stdev(list)
print( "Petal length Stdev:",Petal_Length_Stdev)

list= convertPW(Versicolor)
Petal_Width_Stdev = statistics.stdev(list)
print( "Petal width Stdev:",Petal_Width_Stdev)


Versicolor_Stdev = [Septal_Length_Stdev,Septal_Width_Stdev,Petal_Length_Stdev, Petal_Width_Stdev]
Versicolor_Mean = sum(Setosa)/len(Setosa)

v = lk(X_test,Versicolor_Mean,Versicolor_Stdev)

Liklihood.append(v)
print("")


#Priori
Priori= 1/3
print("Priori probabilities of individual categories:",Priori )
print("")
print("liklihood :")
print(np.matrix(Liklihood))

print("")
a= Posterior(Liklihood,Priori )

print("Posterior :")
print(np.matrix(a))

print("")
#Evaluation
b= Evaluation(a)
print("predicted:",b)
print("actual:",y_test)


t=0
for x in range(len(b)):
    if y_test[x]==b[x]:
        t=t+1
        
print("accuracy:",(t/len(b))*100,"%")        





# In[ ]:





# In[ ]:




