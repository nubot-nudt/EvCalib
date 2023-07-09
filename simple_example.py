#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:00:21 2023

@author: lixiao
"""

import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as sR
import matplotlib.pyplot as plt

class CCA:
    def __init__(self):
        self.a = None
        self.b = None

    def train(self, X, Y, need_R = False):
        Nx, cx = X.shape
        Ny, cy = Y.shape

        X = (X - np.mean(X, 0)) / (np.std(X, 0)+1e-10)
        Y = (Y - np.mean(Y, 0)) / (np.std(Y, 0)+1e-10)

        data = np.concatenate([X, Y], axis = 1)
        cov = np.cov(data, rowvar=False)
        N, C = cov.shape
        Sxx = cov[0:cx, 0:cx]+np.eye(3)*1e-5
        Syy = cov[cx:C, cx:C]+np.eye(3)*1e-5
        Sxy = cov[0:cx, cx:C]+np.eye(3)*1e-5
        Sxx_ = linalg.sqrtm(np.linalg.inv(Sxx))
        Syy_ = linalg.sqrtm(np.linalg.inv(Syy))
        M = Sxx_.T.dot(Sxy.dot(Syy_))
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        u = U[:, 0]
        v = Vt[0, :]
        if need_R:
          M_ = np.linalg.inv(Sxx)@Sxy
          # M_ = np.linalg.inv(Syy)@Sxy.T
          U, S, Vt = np.linalg.svd(M_, full_matrices=False)
          self.R = U@np.array([[1,0,0],
                               [0,1,0],
                               [0,0,np.linalg.det(U@Vt)]])@Vt
        self.a = Sxx_.dot(u)
        self.b = Syy_.dot(v)

    def predict(self, X, Y):
        X_ = X.dot(self.a)
        Y_ = Y.dot(self.b)
        return X_, Y_

    def cal_corrcoef(self, X, Y):
        X_, Y_ = self.predict(X, Y)
        return np.corrcoef(X_, Y_)[0,1]
      


def generate_aRb(indep_rate = 0., R:np.ndarray = np.eye(3), nSamples = 1000, z_scale=1):
  
  # Initialize number of samples
  nSamples = nSamples
  
  # R = np.array([[ 0.5      , -0.8660254,  0.       ],
  #               [ 0.8660254,  0.5      ,  0.       ],
  #               [ 0.       ,  0.       ,  1.       ]])
  indep1 = np.random.randn(nSamples, 3)
  indep2 = np.random.randn(nSamples, 3)
  indep1_2d = np.c_[np.random.randn(nSamples,2), np.zeros(nSamples)]

  # Define three latent variables (number of samples x 1)
  latvar1 = np.random.randn(nSamples,)
  latvar2 = np.random.randn(nSamples,)
  latvar3 = np.random.randn(nSamples,)
  
  latents = np.vstack((latvar1, latvar2, latvar3*z_scale)).T
  
  r_latents = []
  for v in latents:
    r_latents.append(R@v)
  r_latents = np.array(r_latents)

  # R@data1 + indep1 = data2 + indep2
  data1 = latents + indep_rate*indep1_2d
  data2 = r_latents + indep_rate*indep2
  
  return data1, data2


def shift_td(X,Y,td=12):
  '''
  This function generate two relative shift sequences.
  
  T_Y = T_X + td
  
  example:
  a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
  b = a.copy()
  a_t, b_t = shift_td(a,b,3)
  a_t:
  array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17])
  b_t:
  array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
  ===> b_t = a_t + td # td = 3
  
  a_t, b_t = shift_td(a,b,-3)
  a_t: 
  array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
  b_t:
  array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17])
  ===> b_t = a_t + td # td = -3
  
  '''
  offset = td
  if offset < 0:
    X = np.roll(X, offset, axis=0)[:offset]
    Y = Y[:offset]
  else:
    X = np.roll(X, offset, axis=0)[offset:]
    Y = Y[offset:]
  return X, Y



def error(R, gt, data=0):
  # gt = sR.from_euler('zyx', [0, 0, 90], degrees=True).as_matrix()
  return np.arccos((np.trace(R@gt.T)-1)/2)*180/np.pi


def calib_result():
  # print('-------  calib_result: -------')
  
  Rz = np.random.randint(low=-90, high=90)
  Rx = np.random.randint(low=-90, high=90)
  Ry = np.random.randint(low=-90, high=90)
  R_gt = sR.from_euler('zyx', [Rz, Ry, Rx], degrees=True).as_matrix() 
  # R_gt = sR.from_euler('zyx', [0, 0, 90], degrees=True).as_matrix() 
  
  # R_gt = np.array([[ 0.5      , -0.8660254,  0.       ],
  #               [ 0.8660254,  0.5      ,  0.       ],
  #               [ 0.       ,  0.       ,  1.       ]])
  
  X, Y = generate_aRb(0.05, R_gt, nSamples=100)
  # n = X.shape[0]
  clf = CCA()
  clf.train(X, Y, need_R=True)
  # print ('unshifted corr:\t',clf.cal_corrcoef(X, Y))

  td_range = 20
  td_gt = np.random.randint(low=-td_range, high=td_range)


  shifted_X, shifted_Y = shift_td(X, Y, td=td_gt)
  # print ('shifted td corr:\t',clf.cal_corrcoef(shifted_X, shifted_Y))

  max_cor = -10086
  best_guess = 0
  rot = 0
  for guess in range(-td_range,td_range+1):
    X1, Y1 = shift_td(shifted_X, shifted_Y, td=guess)
    clf.train(X1, Y1, need_R=True)
    cor = clf.cal_corrcoef(X1, Y1)  
    if cor > max_cor:
      max_cor = cor
      best_guess = guess
      rot = clf.R
  print('max correlation:\t', max_cor)
  print('best shifted td:\t', best_guess)
  print('Ground truth td:\t', td_gt)
  
  uns_X, uns_Y = shift_td(shifted_X, shifted_Y, td=best_guess)
  
  # uns_td = np.where(X == uns_X[200])[0][0]
  # print(abs(uns_td-200) == abs(best_guess))
  
  stacked_r_list = [ uns_X, uns_Y ]
  H = stacked_r_list[0].T.dot(stacked_r_list[1])
  U, d, Vt = np.linalg.svd(H)
  R_est = Vt.T.dot(U.T)
  
  clf.train(uns_X, uns_Y, need_R=True)
  eR = sR.from_matrix(R_gt).as_euler('zyx', degrees=True)
  eR_est = sR.from_matrix(R_est).as_euler('zyx', degrees=True)
  eRcca = sR.from_matrix(clf.R.T).as_euler('zyx', degrees=True)
  print('Ground truth R: \n', eR)
  print('R_est: \n', eR_est)
  print('R cca:\n', eRcca)
  print('Error R_est: ', error(R_est, R_gt))
  print('Error R_cca: ', error(clf.R.T, R_gt))
  
  print('------------------------\n')
  return np.linalg.norm(eR_est - eR), np.linalg.norm(eRcca - eR)
  


''' main '''

######## test ########

np.set_printoptions(suppress=True)


######################



''' show result '''

calib_result()

''''''''''''''''''





















