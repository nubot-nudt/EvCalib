#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:16:10 2023

@author: lixiao
"""

import numpy as np
from scipy.spatial.transform import Rotation as sR
import scipy
from scipy.interpolate import CubicSpline

def calib_R(va, vb): # calibrate two vector
  H = va.T.dot(vb)
  U, d, Vt = np.linalg.svd(H)
  R = Vt.T@np.array([[1,0,0],
                     [0,1,0],
                     [0,0,np.linalg.det(U@Vt)]])@U.T
  return R

def bundle_rotation_estimation(va, vb, sigma = 0.1, max_iter = 10000):
  '''
  refrence: Real-time rotation estimation for dense depth sensors in piece-wise planar environments
  site: https://ieeexplore.ieee.org/abstract/document/7759355
  '''
  w = np.ones(va.shape[0])
  
  def func(w, v):
    result = []
    for i in range(v.shape[0]):
      result.append(w[i]*v[i])
    return result
  
  R = np.eye(3)

  for i in range(max_iter):
    va = np.apply_along_axis(func, 0, w, va)
    H = va.T.dot(vb)
    U, d, Vt = np.linalg.svd(H)
    R = Vt.T@np.array([[1,0,0],
                       [0,1,0],
                       [0,0,np.linalg.det(U@Vt)]])@U.T

    for j in range(len(w)):
      norm2 = np.linalg.norm(va[j] - vb[j]@R)
      den = np.fmax(sigma, norm2)
      w[j] = 1/den
  return R
    
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
        Sxx_ = scipy.linalg.sqrtm(np.linalg.inv(Sxx))
        Syy_ = scipy.linalg.sqrtm(np.linalg.inv(Syy))
        M = Sxx_.T.dot(Sxy.dot(Syy_))
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        u = U[:, 0]
        v = Vt[0, :]
        if need_R:
          M_ = np.linalg.inv(Syy)@Sxy.T
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


def find_time_shift(ve, ve_ts, vo, vo_ts, td_range):#td_range ms
  interp_func = CubicSpline(vo_ts, vo, axis=0)
  scale = 1000
  max_cor = -10086
  best_guess = 0
  clf = CCA()
  rot = 0
  shifted_vo = 0
  for guess in np.arange(-td_range*scale,(td_range)*scale, 1):
    guess = guess/1000.
    xs = ve_ts + guess
    vo_shift = interp_func(xs)
    clf.train(ve[:,0:3], vo_shift, need_R=True)
    cor = clf.cal_corrcoef(ve, vo_shift)  
    if cor > max_cor:
      max_cor = cor
      best_guess = guess
      rot = clf.R
      shifted_vo = vo_shift
  return best_guess, rot, shifted_vo, max_cor

class CalibrationModel:
  def fit(self, data):
    ve = data[:,0:3]
    vo = data[:,4:7]
    R = calib_R(ve, vo)
    return R
  
  def distance(self, model, point):
    ve = point[0:3]
    vo = point[4:7]
    return np.linalg.norm(ve - vo@model)
  
  def error(self, model, data=0):
    gt = sR.from_euler('zyx', [0, 0, 90], degrees=True).as_matrix()
    return np.arccos((np.trace(model@gt)-1)/2)*180/np.pi

''' test function '''

def re_calib_result(data_file, msg):
  calib_model = CalibrationModel()
  calib_data = np.load(data_file)
  calib_data = calib_data['best_set']
  ve = calib_data[:,0:4]
  vo = calib_data[:,4:8]
  R = calib_R(ve[:,0:3], vo[:,0:3])  # vo[0]@R = ve[0]
  R_euler = sR.from_matrix(R.T).as_euler('zyx', degrees=True)
  print('===== ',msg,' =====')
  print('R_euler: ', R_euler)
  print('error: ', calib_model.error(R))
  print('---\n')
  
def re_cca_result(data_file, msg):
  calib_model = CalibrationModel()
  calib_data = np.load(data_file)
  calib_data = calib_data['best_set']
  ve = calib_data[:,0:4]
  vo = calib_data[:,4:8]
  ve = ve[np.argsort(ve[:,3])]
  vo = vo[np.argsort(vo[:,3])]
  de = np.diff(ve[:,3])
  do = np.diff(vo[:,3])
  ve = ve[1:][(de > 0.) == 1]
  vo = vo[1:][(do > 0.) == 1]
  best_guess, R_cca, vo_shifted, max_cor = find_time_shift(ve[:,0:3], ve[:,3], vo[:,0:3], vo[:,3], 1) # -1000ms - 1000ms
  print('===== ',msg,' =====')
  print('best_guess: ', best_guess)
  print('max_cor: ', max_cor)
  R_shifted = calib_R(ve[:,0:3], vo_shifted[:,0:3])
  R_euler = sR.from_matrix(R_shifted.T).as_euler('zyx', degrees=True)
  print('R_unshift: ', R_euler)
  print('unshift_error: ', calib_model.error(R_shifted))
  
  print('---\n')

def re_bundle_rotation_result(data_file, msg):
  calib_model = CalibrationModel()
  calib_data = np.load(data_file)
  calib_data = calib_data['best_set']
  ve = calib_data[:,0:4]
  vo = calib_data[:,4:8]
  R = bundle_rotation_estimation(ve[:,0:3], vo[:,0:3])  # vo[0]@R = ve[0]
  R_euler = sR.from_matrix(R.T).as_euler('zyx', degrees=True)
  print('===== ',msg,' =====')
  print('R_euler: ', R_euler)
  print('error: ', calib_model.error(R))
  print('---\n')

def re_cca_bundle_rotation_result(data_file, msg):
  calib_model = CalibrationModel()
  calib_data = np.load(data_file)
  calib_data = calib_data['best_set']
  ve = calib_data[:,0:4]
  vo = calib_data[:,4:8]
  ve = ve[np.argsort(ve[:,3])]
  vo = vo[np.argsort(vo[:,3])]
  de = np.diff(ve[:,3])
  do = np.diff(vo[:,3])
  ve = ve[1:][(de > 0.) == 1]
  vo = vo[1:][(do > 0.) == 1]
  best_guess, R_cca, vo_shifted, max_cor = find_time_shift(ve[:,0:3], ve[:,3], vo[:,0:3], vo[:,3], 1) # -1000ms - 1000ms
  print('===== ',msg,' =====')
  print('best_guess: ', best_guess)
  print('max_cor: ', max_cor)
  R_shifted = bundle_rotation_estimation(ve[:,0:3], vo_shifted[:,0:3])
  R_euler = sR.from_matrix(R_shifted.T).as_euler('zyx', degrees=True)
  print('R_unshift_bundle_est: ', R_euler)
  print('unshift_bundle_est_error: ', calib_model.error(R_shifted))
  print('---\n')





''' main '''
np.set_printoptions(suppress=True)
calib_files = ["data/calib_data_time_2traj.npz",
              "data/calib_data_time_3traj.npz",
              "data/calib_data_time_4traj.npz",
              "data/calib_data_time_5traj.npz"]

# VC-1
re_calib_result(calib_files[0], 'vc1_traj_2')
re_calib_result(calib_files[1], 'vc1_traj_3')
re_calib_result(calib_files[2], 'vc1_traj_4')
re_calib_result(calib_files[3], 'vc1_traj_5')

# VC-2
re_cca_result(calib_files[0], 'vc2_traj_2')
re_cca_result(calib_files[1], 'vc2_traj_3')
re_cca_result(calib_files[2], 'vc2_traj_4')
re_cca_result(calib_files[3], 'vc2_traj_5')

# bundle estimation
# re_bundle_rotation_result(calib_files[0], 'bundle_est_traj2')
# re_bundle_rotation_result(calib_files[1], 'bundle_est_traj3')
# re_bundle_rotation_result(calib_files[2], 'bundle_est_traj4')
# re_bundle_rotation_result(calib_files[3], 'bundle_est_traj5')

## VC
re_cca_bundle_rotation_result(calib_files[0], 'vc_bundle_est_traj2')
re_cca_bundle_rotation_result(calib_files[1], 'vc_bundle_est_traj3')
re_cca_bundle_rotation_result(calib_files[2], 'vc_bundle_est_traj4')
re_cca_bundle_rotation_result(calib_files[3], 'vc_bundle_est_traj5')









































