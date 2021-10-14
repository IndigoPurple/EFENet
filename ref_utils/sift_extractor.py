# Created by ytan on 20181208

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os

class SiftExtractor(object):

    def __init__(self, score = 0.3):

        self.score = score
        self.count = 0

    def _get_sift(self,gray):

        '''
        using opencv to get sift feature
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)

        return kp,des

    def _cv_match(self,img_1,kp_1,des_1,img_2,kp_2,des_2):

        bf = cv2.BFMatcher()

        self.count += 1        
        if des_1 is None or des_2 is None:
            return None
        matches = bf.knnMatch(des_1,des_2,k=2)  
        #print (self.count, len(matches))    
        if len(matches) >= 2:
            good_match = []
            for each in matches:
                if len(each) == 2:
                    m,n = each 
                    if m.distance < self.score * n.distance:
                        good_match.append([m])
                else:
                    good_match = None
                    break
            #good_match = [[m] for m, n in matches if m.distance < self.score * n.distance]
        else:
            good_match = None 
         
        #-------------save matched img--------------------
        #img = cv2.drawMatchesKnn(img_1,kp_1,img_2,kp_2,good_match,None,flags=2) 
        #save_dir = './giga_matched_img/'
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)
        #cv2.imwrite('%s%d.png'%(save_dir,self.count),img)
        #cv2.waitKey(1)       

        return good_match

    def get_matched_landmark(self, img_1, img_2):

          
        gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        
        pickle.dump(gray_1,open('img1','wb'))
        pickle.dump(gray_2,open('img2','wb'))

        kp_1,des_1 = self._get_sift(gray_1)
        kp_2,des_2 = self._get_sift(gray_2)
        good_match = self._cv_match(img_1,kp_1,des_1,img_2,kp_2,des_2)
        if good_match is None: 
            return None,None
        
        num_landmark = len(good_match)

        landmark_1 = np.zeros((num_landmark,2), dtype=np.int)
        landmark_2 = np.zeros((num_landmark,2), dtype=np.int)
        for i,each in enumerate(good_match):

            pt_1 = kp_1[each[0].queryIdx].pt
            pt_2 = kp_2[each[0].trainIdx].pt
            landmark_1[i,0] = round(pt_1[0])
            landmark_1[i,1] = round(pt_1[1])
            landmark_2[i,0] = round(pt_2[0])
            landmark_2[i,1] = round(pt_2[1])

        return landmark_1, landmark_2

