testing error, training error
- naive shuffle, mean dummy regressor = 68
- naive shuffle, 1x1x(varlaplace) tilessvr.predict(Xtest), SVR(C=?) = 44
- naive shuffle, 6x3x(varlaplace) tiles, SVR(C=30) = 19.4
- naive shuffle, 12x5x(varlaplace) tiles, SVR(C=30) = 10.3
- naive shuffle, test_size=.150, 26x12x(varlaplace) tiles, SVR(C=.2) = 17.9, 18.1
-    no shuffle, test_size=.150, 26x12x(varlaplace) tiles, SVR(C=.2) = 33.2, 17.7
- chunk shuffle, test_size=.147, 26x12x(varlaplace) tiles, SVR(C=.2) = 39, 16.6
- chunk shuffle, test_size=.147, 26x12x(varlaplace) tiles, SVR(C=.1) = 42, 20
- chunk shuffle, test_size=.147, 26x12x(varlaplace) tiles, SVR(C=.5) = 36, 12
- chunk shuffle, test_size=.147, 14x06x(varlaplace) tiles, SVR(C=.5) = 27, 19
- chunk shuffle, test_size=.147, 14x06x(varlaplace) tiles, SVR(C=.3) = 28.5, 21.5, 39.5
- chunk shuffle, test_size=.245, 14x06x(varlaplace) tiles, SVR(C=.3) = 50, 46, 18
# new crop
- chunk no shuffle, test_size=.245, 14x06x(varlaplace) tiles, SVR(C=.3) = 71, 39, 26.9
- chunk=5 shuffle, test_size=.367, 06x06x(varlaplace) tiles, SVR(C=.3) = 67, 30, 29.1
- chunk=5 shuffle, test_size=.491, 06x06x(varlaplace) tiles, SVR(C=.3) = 67, 30, 29.7
- chunk=5 shuffle, test_size=.73, 06x06x(varlaplace) tiles, SVR(C=.3) = 67, 32, 30.7

# Fixed tiler test_pct=.43                                             
-                                                                 dummy test train test_kf
- chunk=60 focus(4x10), SVR(C=.3) = 70.3 39.6 29.4
- chunk=60 focus(4x10), SVR(C=.3) = 71.6 31.1 22.6
- chunk=60 focus(4x10), test_size=.43, 8x20(varlaplace), gridsearch = 71.6 31.1 22.6

# add gridsearch, cv, standard scaler test_pct=.43  
- chunk=60 focus(3x3),  SVR(C=20, gamma=0.006) 64.0 13.0 2.0
- chunk=60 focus(8x20), SVR(C=10, gamma=0.006) 64.5 13.7 2.4 9.4

# Added kalman filter (still just using focus) test_pct=.43
- nchunks=25 focus(2x4) gridsearch('svr__C':[0001, 001, .01, .1, 1, 10*, 15, 20, 30, 60])  80.7 87.4 56.4 87.2 (cv failing)
- nchunks=25 focus(3x8) gridsearch('svr__C':[0001, 001, .01, .1, 1, 10*, 15, 20, 30, 60]) 74.8 41.1 6.4 33.7
- nchunks=25 focus(6x16) gridsearch('svr__C':[10*, 20, 30, 40, 70, 120]) 65.1 37.4 2.0 31.2 
- nchunks=25 focus(8x20) gridsearch('svr__C':[20, 30, 40, 70, 120]) 75.7 41.2 0.6 37.0
- nchunks=50 focus(8x20) gridsearch('svr__C':[20, 30, 40, 70, 120]) 78.7 25.1 2.6 20.6
- nchunks=50 focus(8x20) SVR(30) 78.7 24.0 1.3 19.3 
- nchunks=50 focus(8x20) RandomForestRegressor() 78.7 18.0 0.2 13.0 
- nchunks=50 focus(8x20) MLP() 78.7 35.2 0.4 18.6
- nchunks=50 focus(8x20) MLP(shuffle=false) 78.7 28.3 3.0 16.0
- nchunks=50 focus(8x20) MLP(shuffle=false, alpha=.01) - - - 15
- nchunks=50 focus(8x20) BaysianRidge() 78.7 40.1 20.0 29.1
- nchunks=50 focus(8x20) EnsembleStack() 78.7 15.6 0.9 11.1

## ideas
- X start chunking
- X increase test_size to at least .25
- X Use df to unshuffle for postprocessing and output
- X accelleration limits based on testing set. This could be a post processing step.concat\
    - Kalman:
      - https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
    - To apply kalman to blur based-must find way to unshuffle.
- X Actually calculate the origin rather than guessing based on the crop.
- X Optical Flow
    - Can we get explicitly always calculate Vr
        - Optical flow techniques for estimation of camera motion parameters insewer closed circuit television inspection videos
          - The reference angle is in the denominator and therefore normalizes the optical flow vector. - Wrong, it eliminates panning but doesn't correct for perspective project. Duh
- X perspective shifting
- X apply perspective shifting/b&w to blur method also
    - speed enhancement for overall extracting
- add debug for tiling (matplotlib subplots imshow)

- stop detection
    - enhance "noise" filter in optical flow by incorporating stdev (problem = slow speeds)
    - structural differenct preprocessor
    - can blur method halp determine "stopped"
- tune optical flow parameters
- tune optical flow preprocessing
- kalman on Vf before training 
- feature extractor
    - X dict of frames model
        - view named by key
    - X can also write final "extract function" against key
- optical flow between more 1 frame delta
    - X need to add "memory" to frame