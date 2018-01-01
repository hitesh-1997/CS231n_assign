import numpy as np
from random import shuffle
from past.builtins import xrange
from math import e

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to un into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dims = W.shape[0]
  classes = W.shape[1]
  scores = X.dot(W)  # SCORE = n*c
  scores -= np.max(scores, axis=1, keepdims=True) 
  scores = np.exp(scores) 
  
  for i in xrange(num_train):
    sum = 0
    for j in range(classes):
      sum+=scores[i][j]
      #print(sum)
    loss+=-np.log(scores[i][y[i]]/sum)
  #print(loss)
  loss/= num_train
  loss+=0.5*reg*np.sum(W*W)
  
  add = np.sum(scores,axis=1)
  scores[np.arange(num_train),y] = add- scores[np.arange(num_train),y]
  scores=scores/add.reshape(num_train,1)
  dW = (X.T).dot(scores)
  dW/=num_train
  dW+=reg*W  
  

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  dims = W.shape[0]
  classes = W.shape[1]
  scores = X.dot(W)  # SCORE = n*c
  scores -= np.max(scores, axis=1, keepdims=True) 
  scores = np.exp(scores) 
  
  new_mat = -np.log(scores[np.arange(num_train),y]/np.sum(scores,axis=1))
  loss = np.sum(new_mat)
  loss/= num_train
  loss+=0.5*reg*np.sum(W*W)
  
  '''
  incorrect implementation of softmaxgradient function:
  correct formula:
  grad L(i) wrt sj = (sscore/total score) 
  grad L(i) wrt si = (sscore/total score-1) 
  '''
  add = np.sum(scores,axis=1)[:,None]
  scores[np.arange(num_train),y] = add- scores[np.arange(num_train),y]
  scores=scores/add.reshape(num_train,1)
  dW = (X.T).dot(scores)
  dW/=num_train
  dW+=reg*W  
  
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

