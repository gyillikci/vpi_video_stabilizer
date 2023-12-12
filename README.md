# vpi_video_stabilizer
VPI based real-time video stabilizer.
This is an initial implementation of https://github.com/ejowerks/wfb-stabilizer by using Jetson and VPI. 

Since the OpenCV consumes alot of CPU power, this repo tries to put the computational burden to GPU instead of CPU by using NVIDIA VPI library. There are still OpenCV functions exist. they will be replaced by GPU based functions
https://www.youtube.com/watch?v=lTuC91WWlQo
