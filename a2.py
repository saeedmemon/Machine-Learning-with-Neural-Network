import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sklearn.discriminant_analysis as da
import sklearn.naive_bayes as bayes
import sklearn.neural_network  as nn
import scipy.stats as ss
import bonnerlib2D as bonner


print('\n')
print('Question 1')
print('---------------------')


#Linear Regression with Feature mapping

print('\n')
print('Question 1(a)')
print('-------------')


#Load in the data
with open('dataA2Q1.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)


def fit_func(x,t,k,size,weight,have_weight):
    
    #weight by default is not used just passed by when we are creating a new y when Weight is already calculated
    
    one_k = np.ones((size,k))*np.arange(1,k+1)
    temp = one_k*x[:,np.newaxis] #makes a matrix multiple of x,2x,3x...,kx for each X
    z_cos = np.cos(temp)
    z_sin = np.sin(temp)
    z = np.c_[np.ones(size),z_cos,z_sin]
    if have_weight == False:
        w = np.linalg.lstsq(z,t,rcond=-1)[0]
        y = np.matmul(z,w)
        return y,w
    
    else:
        y = np.matmul(z,weight)
        return y,weight
    

def fit_plot(dataTrain,dataTest,K):
    
    Xtrain,Ttrain = dataTrain[0],dataTrain[1]
    Xtest,Ttest = dataTest[0],dataTest[1]
    
    train_size = np.shape(Xtrain)[0]
    test_size = np.shape(Xtest)[0]
    
    #fit function
    Ytrain,Weight = fit_func(Xtrain,Ttrain,K,train_size,np.zeros(train_size),False)
    Ytest = fit_func(Xtest,Ttest,K,test_size,np.zeros(train_size),False)[0]

    #error
    Train_err = np.mean(np.square(Ttrain-Ytrain))
    Test_err = np.mean(np.square(Ttest-Ytest))
    
    
    Uplim = np.max(Ttrain)+5
    Lowlim = np.min(Ttrain)-5
    
    new_X = np.linspace(np.min(Xtrain),np.max(Xtrain),num=1000)
    new_Y = fit_func(new_X,Ttrain,K,1000,Weight,True)[0]

    plt.scatter(Xtrain,Ttrain,s=20,c='b')
    plt.plot(new_X,new_Y,c='r')
    plt.ylim((Lowlim,Uplim))

    
    return Weight,Train_err,Test_err
    


print('\n')
print('Question 1(b)')
print('-------------')

def show_plot(k,q):
    Weight,Train_err,Test_err = fit_plot(dataTrain,dataTest,k)
    plt.title('Question 1('+q+'): the fitted function (K= '+str(k)+')')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print(k,Weight,Train_err,Test_err)

show_plot(3,'b')

print('\n')
print('Question 1(c)')
print('-------------')


show_plot(9,'c')


print('\n')
print('Question 1(d)')
print('-------------')

show_plot(12,'d')


print('\n')
print('Question 1(e)')
print('-------------')
i=1
plt.figure(figsize=(5,5))
           
while i < 13: 
    plt.subplot(4,3,i)
    fit_plot(dataTrain,dataTest,i)
    plt.suptitle('Question 1(e): fitted functions for many values of K.')
    i+=1
plt.show()


print('\n')
print('Question 1(f)')
print('-------------')


print('I dont know')






print('\n')
print('Question 2')
print('---------------------')

#Probalistic Multi-class Classification

print('\n')
print('Question 2(a)')
print('-------------')

#Load and visualize the data

with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    
Xtrain,Ttrain = dataTrain
Xtest,Ttest = dataTest


def plot_data(X,T):
    
    xmax = np.max(X) + 0.1 
    xmin = np.min(X) - 0.1
    

    colors = np.array(['r','b','g']) 
    plt.scatter(X[:,0],X[:,1],color = colors[T],s=2)
    plt.xlim(xmin,xmax)
    plt.ylim(xmin,xmax)



print('\n')
print('Question 2(b)')
print('-------------')

#Implemented Logistic regression to fit the training data

clf = lin.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain,Ttrain)

def accuracyLR(clf,X,T):
    
    w = clf.coef_
    bias = clf.intercept_
    z = np.matmul(X,w.T) + bias
    row_max = np.amax(z,1)
    row, col = np.where(z == row_max[:,None]) 
    accuracy2 = np.mean(col == Ttest)
    return accuracy2


#Used score method to test how good the model fits the data using Test data
accuracy1 = clf.score(Xtest,Ttest)
accuracy2 = accuracyLR(clf,Xtest,Ttest)

print("This is accuracy 1 ", accuracy1)
print('This is accuracy 2 ', accuracy2)
print('The difference is ', accuracy1-accuracy2)

plt.figure()
plot_data(Xtrain,Ttrain)
plt.title('Question 2(b): decision boundaries for logistic regression.')
bonner.boundaries(clf)


print('\n')
print('Question 2(c)')
print('-------------')


#Implemented quardratic Discriminative analysis model to fit the training data
clf1 = da.QuadraticDiscriminantAnalysis(store_covariance=True)
clf1.fit(Xtrain,Ttrain)

#This function is implemented in attempt to better understand the mathmatics and statistics built behind the score method.
def accuracyQDA(clf,X,T):
        
    mean = clf1.means_
    cov = clf1.covariance_
    class_prior = clf1.priors_

    i = 0
    predict = []

    while i < np.shape(class_prior)[0] :

        predi = ss.multivariate_normal.pdf(X,mean[i],cov[i])*class_prior[i]
        predict.append(predi)
        i+=1   
        
    correct = np.argmax(predict,0)
    return np.mean(correct==T)     

#Test the model using test data.
#Comparison between the two accuracy, Idealy the difference should be close to zero
accuracy1 = clf1.score(Xtest,Ttest)  
accuracy2 = accuracyQDA(clf1,Xtest,Ttest)

print("This is accuracy 1 ", accuracy1)
print('This is accuracy 2 ', accuracy2)
print('The difference is ', accuracy1-accuracy2)

plt.figure()
plot_data(Xtrain,Ttrain)
plt.title('Question 2(c): decision boundaries for quadratic discriminant analysis.')
bonner.boundaries(clf1)



print('\n')
print('Question 2(d)')
print('-------------')

#Implemented Gaussian Naive Bayes model to fit the training data
clf2 = bayes.GaussianNB()
clf2.fit(Xtrain,Ttrain)

def accuracyNB(clf,X,T):
    
    mean = clf.theta_
    var = clf.sigma_
    prior = clf.class_prior_
    devi = np.sqrt(var)


    i,j = np.shape(X)
    X = np.reshape(X,[i,1,j])

    numer = np.exp(-np.square(X-mean)/(2*var))
    denom = math.sqrt(2*math.pi)*devi
    
    Xi_given_k = np.divide(numer,denom)
    product = np.prod(Xi_given_k,2)

    
    predict = product * prior
    correct = np.argmax(predict,1)

    return np.mean(correct == T)
    
    

#Evaluate the how good the model fits the data
accuracy1 = clf2.score(Xtest,Ttest)    
accuracy2 = accuracyNB(clf2,Xtest,Ttest)

print("This is accuracy 1 ", accuracy1)
print('This is accuracy 2 ', accuracy2)
print('The difference is ', accuracy1-accuracy2)


plt.figure()
plot_data(Xtrain,Ttrain)
plt.title('Question 2(d): decision boundaries for Gaussian naive Bayes.')
bonner.boundaries(clf2)


print('\n')
print('Question 3')
print('---------------------')

#Neural Network: basics

print('\n')
print('Question 3(a)')
print('-------------')

#Reload the data
with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    
Xtrain,Ttrain = dataTrain
Xtest,Ttest = dataTest




print('\n')
print('Question 3(b)')
print('-------------')

#Set random seed to recieve more consistent accuracys to compare work.
np.random.seed(0)

#Implemented Neural Network Multi layer Perceptron classifier using logistic regression and stochastic gradient descent
clf = nn.MLPClassifier(hidden_layer_sizes=(1,),
                       activation='logistic',
                       solver='sgd',
                       max_iter=1000,
                       learning_rate_init=0.01,
                       tol = int(10**-6))

def plot_nn(clf,X,T,question,unit):
    clf.fit(Xtrain,Ttrain)

    accuracy = clf.score(X,T)
    print('The is the accuracy ',accuracy)
    
    
    plot_data(Xtest,Ttest)
    plt.title('Question 3('+question+'): Neural net with '+unit+' hidden unit.')
    bonner.boundaries(clf)
 
plt.figure()    
plot_nn(clf,Xtrain,Ttrain,'b',str(1))


print('\n')
print('Question 3(c)')
print('-------------')

np.random.seed(0)

#Similar step from above but with 2 hidden layers
clf2 = nn.MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        solver='sgd',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        tol = int(10**-6))

plt.figure()
plot_nn(clf2,Xtrain,Ttrain,'c',str(2))

print('\n')
print('Question 3(d)')
print('-------------')

np.random.seed(0)

#Same method but with 9 layers
clf3 = nn.MLPClassifier(hidden_layer_sizes=(9,),
                        activation='logistic',
                        solver='sgd',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        tol = int(10**-6))

plt.figure()
plot_nn(clf3,Xtrain,Ttrain,'d',str(9))
plt.show()


print('\n')
print('Question 3(e)')
print('-------------')

# Here we have the hidden layer fixed but loop through data with same model with different training iteration.
#The iteration goes from 2^2 to 2^10 and plots the each indivdual model to compare the difference
i=1
plt.figure()
while i<=9:
    plt.subplot(3,3,i)
    np.random.seed(0)

    clf4 = nn.MLPClassifier(hidden_layer_sizes=(7,),
                            activation='logistic',
                            solver='sgd',
                            max_iter=int(2**(1+i)),
                            learning_rate_init=0.01,tol = int(10**-6))
    clf4.fit(Xtrain,Ttrain)


    accuracy = clf4.score(Xtest,Ttest)
    print('The is the accuracy ',accuracy)
    plot_data(Xtrain,Ttrain)
    bonner.boundaries(clf4)
    i+=1
plt.suptitle('Question 3(e): different numbers of epochs.')

print('\n')
print('Question 3(f)')
print('-------------')


#Fixed model with different intial weight.
i=1
plt.figure().canvas.manager.full_screen_toggle()
while i<=9:
    plt.subplot(3,3,i)
    np.random.seed(i)
    clf5 = nn.MLPClassifier(hidden_layer_sizes=(5,),
                            activation='logistic',
                            solver='sgd',
                            max_iter=1000,
                            learning_rate_init=0.01,
                            tol = int(10**-6))
    clf5.fit(Xtrain,Ttrain)
    
    accuracy = clf5.score(Xtest,Ttest)
    print('The is the accuracy ',accuracy)
    plot_data(Xtrain,Ttrain)
    bonner.boundaries(clf5)
    i+=1
    
    
plt.suptitle('Question 3(f): different initial weights.')
plt.show()



print('\n')
print('Question 3(g)')
print('-------------')

np.random.seed(0)

#Implemented MLP classifier to training data
clf6 = nn.MLPClassifier(hidden_layer_sizes=(9,),
                        activation='logistic',
                        solver='sgd',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        tol = 10**-6
                        )

clf6.fit(Xtrain,Ttrain)


#Re-build the score method using math and stats
def accuracyNN(clf,X,T):
    
    w = clf.coefs_
    bias = clf.intercepts_
    
    z = np.matmul(X,w[0]) + bias[0]
    y = 1/(1+np.exp(-z))
    z1 = np.matmul(y,w[1]) + bias[1]
    predict = np.argmax(z1,1)
    

    return np.mean(predict == T)
    

#Test the difference, idealy the difference should be close to zero
accuracy1 = clf6.score(Xtest,Ttest)   
accuracy2 = accuracyNN(clf6,Xtest,Ttest)  
    
print("This is accuracy 1 ", accuracy1)
print('This is accuracy 2 ', accuracy2)
print('The difference is ', accuracy1-accuracy2)
        
    
    
print('\n')
print('Question 3(h)')
print('-------------')   


#Implementing the mean cross entropy of the neural network on test data
def ceNN(clf,X,T):
    
    #weight and bias
    w = clf.coefs_
    bias = clf.intercepts_
    
    #Transform and filter data
    classes = np.max(T)+1
    T_size = np.shape(T)[0]
    one_hot = np.tile(np.arange(classes),(T_size,1))
    
    zeros = np.zeros((T_size,classes))
    t = np.where(one_hot==T[:,None],1,0) 
    
    #First Cross entropy from library 
    CE1_prob= clf.predict_log_proba(X)
    CE1 = np.sum(-t*CE1_prob)/T_size

    #Second Cross entropy is directly implemented from fundementals
    z = np.matmul(X,w[0]) + bias[0]
    y = 1/(1+np.exp(-z))
    z1 = np.matmul(y,w[1]) + bias[1]
    y_softmax = np.exp(z1)/np.sum(np.exp(z1),1)[:,None]
    CE2 = np.sum(-t*np.log(y_softmax))/T_size
        
    return CE1,CE2


np.random.seed(0)

clf7 = nn.MLPClassifier(hidden_layer_sizes=(9,),
                        activation='logistic',
                        solver='sgd',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        tol = 10**-6
                        )

clf7.fit(Xtrain,Ttrain)

#Comparison between the two Cross entropys
CE1,CE2 = ceNN(clf7,Xtest,Ttest)
print('This is cross entropy 1  ',CE1)
print('This is cross entropy 2  ',CE2)
print('This is the difference  ',np.mean(CE1-CE2))    
    

#Question 4 was a proof question written in latex.


print('\n')
print('Question 5')
print('---------------------')

#Neural Network Implementation on image recognition 

print('Question 5(a)')
print('-------------')   

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
    

def index(d1,d2):
    
    index_train = (Ttrain==d1) | (Ttrain==d2) 
    index_test = (Ttest==d1) | (Ttest==d2) 
    
    x_train = Xtrain[index_train]
    t_train = Ttrain[index_train]
    x_test = Xtest[index_test]
    t_test = Ttest[index_test]
    
    t_train = np.where(t_train==d1,1,0)
    t_test = np.where(t_test==d1,1,0)
    
    
    return x_train,t_train,x_test,t_test




print('Question 5(b)')
print('-------------')   

#Function that evaluates classfier clf on data X,T and returns the accuracy and cross entropy
def evaluateNN(clf,X,T):
    
    w = clf.coefs_
    bias = clf.intercepts_
    
    x_c = np.matmul(X,w[0]) + bias[0]
    h = np.tanh(x_c)
    h_c = np.matmul(h,w[1]) + bias[1]
    g = np.tanh(h_c)
    g_c = np.matmul(g,w[2]) + bias[2]

    output = 1/(1+np.exp(-g_c))
    output1 =  np.concatenate((np.round(output))).astype(int)   #for accuracy

    classes = 2
    T_size = np.shape(T)[0]
    one_hot = np.tile(np.arange(classes),(T_size,1))  #repeats 0, 1  T_size times
    
    zeros = np.zeros((T_size,classes))
    t = np.where(one_hot==T[:,None],1,0) 
    

    
    accuracy1 = clf.score(X,T) 
    accuracy2 = np.mean(output1 == T)
    
    
    CE1_prob = clf.predict_proba(X)
    CE1 =  np.mean(-t*np.log(CE1_prob) -(1-t)*np.log(1-CE1_prob))

    CE2_prob = np.c_[1-output,output]
    CE2 =  np.mean(-t*np.log(CE2_prob) -(1-t)*np.log(1-CE2_prob))
    
    
    return accuracy1,accuracy2,CE1,CE2


print('Question 5(c)')
print('-------------')   

#Now we use function above to test different classifiers

#Here we fit classfier with learning rate = 0.01, batch size 100 and maximum 100 iteration of training.
np.random.seed(0)
clf = nn.MLPClassifier(hidden_layer_sizes=(100,100,),
                        activation='tanh',
                        solver='sgd',
                        max_iter=100,
                        learning_rate_init=0.01,
                        batch_size=100,
                        tol = 10**-6,
                        # verbose=True,
                        )

x_train,t_train,x_test,t_test = index(5,6) 
# x_train,t_train,x_test,t_test = index(4,5)

clf.fit(x_train,t_train)


#Evaluate the classfier
a1,a2,ce1,ce2 = evaluateNN(clf,x_test,t_test)


print("This is accuracy 1 ", a1)
print('This is accuracy 2 ', a2)
print('The difference is ', a1-a2)


print('This is cross entropy 1  ',ce1)
print('This is cross entropy 2  ',ce2)
print('This is the difference  ',ce1-ce2)    
         


print('Question 5(d)')
print('-------------')   

#Make an interation on increment of batch_size to see how does the accurcy change.
i = 0
accuracy = []
CE = []
batch_size = []

x_train,t_train,x_test,t_test = index(5,6)
# x_train,t_train,x_test,t_test = index(4,5)
while i<14:
    np.random.seed(0)
    clf = nn.MLPClassifier(hidden_layer_sizes=(100,100,),
                        activation='tanh',
                        solver='sgd',
                        max_iter=1,
                        learning_rate_init=0.001,
                        batch_size=2**i,
                        tol = 10**-6,
                        # verbose=True,
                        )
    clf.fit(x_train,t_train)
    a1,a2,ce1,ce2 = evaluateNN(clf,x_test,t_test)
    accuracy.append(a2)
    CE.append(ce2)
    batch_size.append(2**i)
    i+=1
    
plt.plot(batch_size,accuracy)
plt.xscale('log')
plt.title('Question 5(d): Accuracy v.s. batch size')
plt.show()

plt.plot(batch_size,CE)
plt.xscale('log')
plt.title('Question 5(d): Cross entropy v.s. batch size')
plt.show()


