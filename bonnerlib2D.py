#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:01:18 2017

@author: anthonybonner
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Plot the decision boundaries of classifier clf
# and colour the decision regions.
# Assume three classes.
def boundaries(clf):
    
    # The extent of xy space
    ax = plt.gca()
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
     
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision function at the grid points
    Z = clf.predict(mesh)
    
    # Use pale colours for the decision regions.
    Z = Z.reshape(xx.shape)
    mylevels=[-0.5,0.5,1.5,2.5]
    ax.contourf(xx, yy, Z, levels=mylevels, colors=('red','blue','green'), alpha=0.2)
    
    # draw the decision boundaries in solid black
    ax.contour(xx, yy, Z, levels=3, colors='k', linestyles='solid')
    
        


# Plot the decision function of classifier clf in 3D.
# if Cflag=1 (the default), draw a contour plot of the decision function
# beneath the 3D plot.

def df3D(clf,cFlag=1):    
    
    # the extent of xy space
    ax = plt.gca()
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.01    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision function at the grid points
    Z = clf.predict_proba(mesh)[:,1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    ax.plot_surface(xx, yy, Z, cmap=cm.RdBu, linewidth=0, rcount=75, ccount=75)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contour(xx,yy,Z,levels=mylevels,linestyles='solid',linewidths=3,cmap=cm.RdBu)
    
    # limits of the vertical axis
    z_min = 0.0
    z_max = 1.0
    
    if cFlag == 1:
        # display a contour plot of the decision function
        z_min = -0.5
        ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, offset=z_min)
    
    ax.set_zlim(z_min,z_max)
    