import cv2
import numpy as np
import matplotlib.pyplot as plt
from filter import detect, linear_correlation
from filter import unblur1, unblur2

class LinearTracker:
    def __init__(self, sigma, lambda_val, interp_factor, mode, output_size=(64, 64)):
        self.sigma = sigma
        self.lambda_val = lambda_val
        self.interp_factor = interp_factor
        self.output_size = output_size # Feel free to mess around with this parameter to increase/decrease performance/accuracy
        self.mode = mode
        self.correlation_function = linear_correlation
        self.w = None
        self.x = None
        self.roi = None

    # Extract and preprocess image patch.
    def preprocess_patch(self, image, roi, output_size):
        x, y, w, h = map(int, roi)
        patch = cv2.getRectSubPix(image, (w, h), (x + w/2, y + h/2))
        patch = cv2.resize(patch, output_size) # resize to help with frame rate
        patch = patch.astype(np.float32) / 255
        patch = patch - np.mean(patch)

        patch = patch.sum(axis=2)
        if self.mode == 1:
            patch = unblur1(patch)
        elif self.mode ==2:
            patch = unblur2(patch)
        return patch
    
    def init(self, frame, roi):
        self.roi = roi
        self.x = self.preprocess_patch(frame, roi, self.output_size)
        self.w = self.correlation_function(self.x, self.sigma, self.lambda_val)
    
    def update(self, frame):
        if self.roi is None:
            return None
        
        z = self.preprocess_patch(frame, self.roi, self.output_size)

        v_max, h_max = detect(self.w, z)
        v_max = v_max - self.output_size[0] if v_max > self.output_size[0]//2 else v_max
        h_max = h_max - self.output_size[0] if h_max > self.output_size[0]//2 else h_max
        
        dx = h_max * self.roi[2] / self.output_size[0]
        dy = v_max * self.roi[3] / self.output_size[1]
        
        self.roi = (
            self.roi[0] + dx,
            self.roi[1] + dy,
            self.roi[2],
            self.roi[3]
        )

        new_x = self.preprocess_patch(frame, self.roi, self.output_size)
        new_w = self.correlation_function(self.x, self.sigma, self.lambda_val)

        self.x = (1 - self.interp_factor) * self.x + self.interp_factor * new_x
        self.w = (1 - self.interp_factor) * self.w + self.interp_factor * new_w

        return self.roi, z