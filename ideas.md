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


# lk_flow all BW, All perspective corrected scaled lk_kf can predict 10.3
- dummy test train test_kf
- 62.7 18.8 20.8 8.0 - nchunks=50 test_pct=.43 focus(1x3) lk(?) SVR(?)                  
- 62.7 15.4 9.8 5.2 - EmsembleStack(svr(c=20), rf, mlp)                                       
- 66.3 20.9 9.1 9.0 - test_pct=.70                                           
- 66.3 20.9 10.8 9.3 - EmsembleStack(svr(c=.3), rf, mlp)                                                    
- 62.7 15.8 10.4 5.5 - test_pct=.43                                                
- 62.7 16.0 8.7 5.3 - EmsembleStack(svr(c=.3), rf)
- 62.7 16.0 8.7 5.3 - move zero clamp to after kalman
- 62.7 16.0 8.0 5.0 - use Pipe(StdScale,Ensemble(svr(c=.3), rf))
soley based on direct linear optical flow we can estimate 10.3, with 3 frame multi-flow 10.2 (scaled on all data, cheating..)
dummy	test	train	test_kf	NoML     NoML is best we can get without any fitting e.g. the final 0_kf or 024_kf scaled
XXXXX	16.0	8.00	5.00	10.3	single frame optical no focusbase (include stdev?)
XXXXX	10.3	10.8	4.70	10.2	3 frame optical, 024_kf added to others
XXXXX	15.9	4.60	9.10	10.2	last + 3 focus features, ...hmmm maby need more model complexity
79.1	15.6	4.20	9.00	10.2	pretty dissapppoint gridsearch..C=20, n_estimators=150, looking back last round was already overfit..
62.6	11.2	11.4	4.40	10.2	ensemble(svr(c=.3,), rf(n_estimators=60)) Index([0, 2, 3, 4, '024'], dtype='object')
78.2	13.1	10.9	6.30	10.2	above with removed std(X3), guess it's needed
ok gaussian is really good
78.8	22.7	19.7	13.5	7.2		7x7 Gauss made 1-frame optical fundamentally better
***added gaussian7x7 to main notebook
***Found "absurd" cap likely being hit on high side (140) when using 3 frame diff
***incorporated extractor enhancements

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
    - X speed enhancement for overall extracting
- X add debug for tiling (matplotlib subplots imshow)

- stop detection
    - X enhance "noise" filter in optical flow by incorporating stdev (problem = slow speeds)
    - X can blur method halp determine "stopped"
   
- Feature Extractor
    - X dict of frames model
        - X view named by key
    - X can also write final "extract function" against key
    - names for headers on X df (from via_xxxxx)
    - Xprocess via (done in gauss_sobel)
    - Xfe.add_step(analyze_frames(...)) --> fe.add_analyzer(...)
    - Xfe.add_step(process_frames(via=lambda img: cv2.GaussianBlur(img,(7,7),0))) --> fe.add_processcor(lambda img: cv2.GaussianBlur(img,(7,7),0)))
    - Xfe.add_step --> fe.add_filteragg

- Processing
    - Pre-extract
        - Xsharpen,blur, etc before lk optical flow
            - X sobel filter?...NO
            - XGaussian....yeessir!!
    - Post-extract
        - <>kalman on Vf before training...too risky, how would that affect training
        - <>0MPH clamp before training ...on what? raw Vf?
- Feature Extraction
    - Optical flow between more 1 frame delta
        - X need to add "memory" to frame ...done, now have framework for extraction pipeline also
        - X how to combine (ml or within single feature?)...as outgoing feature, math to combine into 023kf
        - X maybe use same corners for both and see which are correlated....too much work
    
    - X ratio of good to bad vectors for lk (STOP DETECT) ....didn't really work
    - difference between frame (STOP DETECT)....not tried yet
- Tuning
    - tune optical flow parameters
        - corner quality
    - tune ML model
        - really want to prevent overfitting
    - kf tuning
    - Tweak projection transform matrix
- todo
    - Xincorp process via into speed from gauss_sobel
    - build actual system to read in test video and do inference
    - Xtest gauss using regression (using gaussian go)