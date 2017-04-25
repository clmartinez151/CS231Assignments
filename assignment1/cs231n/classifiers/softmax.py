import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    # Normalization trick to avoid numerical instability, http://cs231n.github.io/linear-classify/#softmax
    scores -= np.max(scores)
    
    correct_class_score = scores[y[i]]
    fj = 0.0
    
    for j in range(num_classes):
      fj += np.exp(scores[j])
    
    for j in range(num_classes):
      dscore = np.exp(scores[j])/fj
      dW[:, j] += (dscore-(j==y[i]) )*X[i]
    
    loss += np.log(fj) - correct_class_score
  
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW += reg*W

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
  num_train = X.shape[0] #based on http://cs231n.github.io/neural-networks-case-study/
  scores = np.dot(X, W)

  # get unnormalized probabilities
  exp_scores = np.exp(scores)

  # normalize them for each example
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  corect_logprobs = -np.log(probs[range(num_train),y])
  
  # compute the loss: average cross-entropy loss and regularization
  data_loss = np.sum(corect_logprobs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train

  dW = np.dot(X.T, dscores)
  dW += reg*W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

