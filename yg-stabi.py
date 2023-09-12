#!/usr/bin/python3
# Author: gyillikci
# Thanks to https://github.com/ejowerks/wfb-stabilizer

import cv2
import numpy as np
import sys
import vpi
# Usage: python ejo_wfb_stabilizer.py [optional video file]
# press "Q" to quit

#################### USER VARS ######################################
# Decreases stabilization latency at the expense of accuracy. Set to 1 if no downsamping is desired. 
# Example: downSample = 0.5 is half resolution and runs faster but gets jittery
downSample = 1.0

#Zoom in so you don't see the frame bouncing around. zoomFactor = 1 for no zoom
zoomFactor = 1.0

# pV and mV can be increased for more smoothing #### start with pV = 0.01 and mV = 2 
processVar=0.03
measVar=2

# set to 1 to display full screen -- doesn't actually go full screen if your monitor rez is higher than stream rez which it probably is. TODO: monitor resolution detection
showFullScreen = 1

# If test video plays too fast then increase this until it looks close enough. Varies with hardware. 
# LEAVE AT 1 if streaming live video from WFB (unless you like a delay in your stream for some weird reason)
delay_time = 1 


######################## Region of Interest (ROI) ###############################
# This is the portion of the frame actually being processed. Smaller ROI = faster processing = less latency
#
# roiDiv = ROI size divisor. Minimum functional divisor is about 3.0 at 720p input. 4.0 is best for solid stabilization.
# Higher FPS and lower resolution can go higher in ROI (and probably should)
# Set showrectROI and/or showUnstabilized vars to = 1 to see the area being processed. On slower PC's 3 might be required if 720p input
roiDiv = 8.0

# set to 1 to show the ROI rectangle 
showrectROI = 0

#showTrackingPoints # show tracking points found in frame. Useful to turn this on for troubleshooting or just for funzies. 
showTrackingPoints = 0

# set to 1 to show unstabilized B&W ROI in a window
showUnstabilized = 1

# maskFrame # Wide angle camera with stabilization warps things at extreme edges of frame. This helps mask them without zoom. 
# Feels more like a windshield. Set to 0 to disable or find the variable down in the code to adjust size of mask
maskFrame = 1

######################## Video Source ###############################

# Your stream source. Requires gstreamer libraries 
# Can be local or another source like a GS RPi
# Check the docs for your wifibroadcast variant and/or the Googles to figure out what to do. 

# Below should work on most PC's with gstreamer  -- ###  #### #### Without hardware acceleration you may need to reduce your stream to 920x540 ish #### #### ###
SRC = 'udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false'

# Below is for author's Ubuntu PC with nvidia/cuda stuff running WFB-NG locally (no groundstation RPi). Requires a lot of fiddling around compiling opencv w/ cuda support
#SRC = 'udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay !  h264parse ! nvh264dec ! videoconvert ! appsink sync=false'

######################################################################

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
	display_width = 1280,
    display_height = 720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
lk_params = dict( winSize  = (15,15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
count = 0
a = 0
x = 0
y = 0
Q = np.array([[processVar]*3])
R = np.array([[measVar]*3])
K_collect = []
P_collect = []
prevFrame = None

# open local video file, warning no filetype validation 
if len(sys.argv) == 2:
	SRC=sys.argv[1]

video = cv2.VideoCapture(SRC)#gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #1080, 720, 360
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1920, 1280, 640
backend = vpi.Backend.CUDA
while True:
	grab, frame = video.read()
	if grab is not True:
		exit() 
	res_w_orig = frame.shape[1]
	res_h_orig = frame.shape[0]
	res_w = int(res_w_orig * downSample)
	res_h = int(res_h_orig * downSample)
	top_left= [int(res_h/roiDiv),int(res_w/roiDiv)]
	bottom_right = [int(res_h - (res_h/roiDiv)),int(res_w - (res_w/roiDiv))]
	frameSize=(res_w,res_h)
	Orig = frame
	if downSample != 1:
		frame = cv2.resize(frame, frameSize) # downSample if applicable
	currFrame = frame
	currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
	currGray = currGray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]  ] #select ROI

	if prevFrame is None:
		prevOrig = frame
		prevFrame = frame
		prevGray = currGray
	
	if (grab == True) & (prevFrame is not None):
		if showrectROI == 1:
			cv2.rectangle(prevOrig,(top_left[1],top_left[0]),(bottom_right[1],bottom_right[0]),color = (211,211,211),thickness = 1)
		# Not in use, save for later
		#gfftmask = np.zeros_like(currGray)
		#gfftmask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 255
		with vpi.Backend.CUDA:
			vpi_frame = vpi.asimage(frame, vpi.Format.BGR8).convert(vpi.Format.S16)
		with vpi.Backend.PVA:
			curFeatures, scores = vpi_frame.harriscorners(strength=0.1, sensitivity=0.01)
			print(curFeatures.size)
		with curFeatures.lock_cpu() as featData, scores.rlock_cpu() as scoresData:
			prevPts = np.array([[p[0], p[1]] for p in featData]).reshape(-1, 1, 2)
			print(prevPts.size)
		#prevPts = cv2.goodFeaturesToTrack(prevGray,maxCorners=400,qualityLevel=0.01,minDistance=30,blockSize=3)
		if prevPts is not None:
			try:
				currPts, status, err = cv2.calcOpticalFlowPyrLK(prevGray,currGray,prevPts,None,**lk_params)
			except:
				print(err)
			if prevPts.shape == currPts.shape:
				idx = np.where(status == 1)[0]
				prevPts = prevPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)]) 
				currPts = currPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)])
			if showTrackingPoints == 1:
				for pT in prevPts:
					cv2.circle(prevOrig, (int(pT[0][0]),int(pT[0][1])) ,5,(211,211,211))
			if prevPts.size & currPts.size:
				m, inliers = cv2.estimateAffinePartial2D(prevPts, currPts)
			if m is None:
				m = lastRigidTransform
			# Smoothing
			dx = m[0, 2]
			dy = m[1, 2]
			da = np.arctan2(m[1, 0], m[0, 0])
		else:
			dx = 0
			dy = 0
			da = 0

		x += dx
		y += dy
		a += da
		Z = np.array([[x, y, a]], dtype="float")
		if count == 0:
			X_estimate = np.zeros((1,3), dtype="float")
			P_estimate = np.ones((1,3), dtype="float")
		else:
			X_predict = X_estimate
			P_predict = P_estimate + Q
			K = P_predict / (P_predict + R)
			X_estimate = X_predict + K * (Z - X_predict)
			P_estimate = (np.ones((1,3), dtype="float") - K) * P_predict
			K_collect.append(K)
			P_collect.append(P_estimate)
		diff_x = X_estimate[0,0] - x
		diff_y = X_estimate[0,1] - y
		diff_a = X_estimate[0,2] - a
		dx += diff_x
		dy += diff_y
		da += diff_a
		m = np.zeros((2,3), dtype="float")
		m[0,0] = np.cos(da)
		m[0,1] = -np.sin(da)
		m[1,0] = np.sin(da)
		m[1,1] = np.cos(da)
		m[0,2] = dx
		m[1,2] = dy
		perspective = np.identity(3)
		perspective[0,0] = np.cos(da)
		perspective[0,1] = -np.sin(da)
		perspective[0, 2] = 0
		perspective[1,0] = np.sin(da)
		perspective[1,1] = np.cos(da)
		perspective[1, 2] = 0
		perspective[0,2] = dx
		perspective[1,2] = dy
		with backend:
			vpi_frame = vpi.asimage(prevOrig, vpi.Format.BGR8)
			vpi_frame = vpi_frame.perspwarp(perspective)
			with vpi_frame.rlock_cpu() as data:
				cv2.imshow("deneme", data)
		
		if showUnstabilized == 1:
			cv2.imshow("Unstabilized ROI",prevGray)
		if cv2.waitKey(delay_time) & 0xFF == ord('q'):
			break
		
		
		prevOrig = Orig
		prevGray = currGray
		prevFrame = currFrame
		lastRigidTransform = m
		count += 1
	else:
		exit()
 
video.release()
 
cv2.destroyAllWindows()
