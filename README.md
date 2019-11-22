This is an implementation of simple feedforward networks. The main goal was to play around with
the training of the network's capactiy and learning rate and observe the performance. 
All the models were abel to achieve above 99% accuracy.

Plots with decision boundary shows the performance of the moidel. 
Exponential decay was also implemented on models in FF_ellipse.py, FF_hexa.py, FF_star.py to minimize the loss

This directory contains the following:
        XORNN.py -> Which contains the code for problem 1. The main method has comments that contains commands for running 
        
        FF_ellipse.py -> feedforward that is used for training on the ellipse data. You can simply run this
        and excute. Training will start from the saved model in models. If you want to train from scratch, 
        set the load variable to False in the main method.
        
        FF_Hexa.py -> feedforward that is used for training on the Hexagon data. You can simply run this
        and excute. Training will start from the saved model in models. If you want to train from scratch, 
        set the load variable to False in the main method.
        
        FF_star.py -> feedforward that is used for training on the star data. You can simply run this
        and excute. Training will start from the saved model in models. If you want to train from scratch, 
        set the load variable to False in the main method.
        
        plots -> directory that contains plots from all the models for this project. The scripts
        may overwrite the plots 
        
        models -> Directory that contains the trained models for each of the problems in this prohject. They
        are named in the following way.