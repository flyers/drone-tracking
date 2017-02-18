#!/usr/bin/env python
import os
import Image
import sys
import re

#img =Image.open(r"/data/nwangab/n01440764/n01440764_1232.JPEG")
 #img=Image.open("page.jpeg")
#width=img.size[0]
#height=img.size[1]
#if width>height:
#	newHeight=256
#	newWidth=256*width/height
#else:
#	newWidth=256
#	newHeight=256*height/width
#newImg=img.resize((newWidth,newHeight),Image.ANTIALIAS)
 #newImg.save("page.jpeg")
#newImg.save("/data/nwangab/n01440764/n01440764_1232.JPEG")

log=open('resize.log','w+')
#log.write("x")
#log.close()
inputDir='/data/kxmo/ImageNet/train_extract/'
outputDir=''
#folds=os.listdir(dir)
#for fold in folds: 
#	print fold
	#continue
	#log=open('resize.log','w')
#	log.write(fold+"\n")
	#log.close()
	#sys.exit()
#	files=os.listdir(os.path.join(dir,fold))
files=os.listdir(inputDir)
for file in files:
	picPath=os.path.join(inputDir,file)
		#print picPath
	log.write(picPath+"\n")
		#pic=Image.open(os.path.join(dir,fold,file))
	try:
		img=Image.open(picPath)
		preImg=Image.open(os.path.join(outputDir,file))
	except IOError:
		pass
	preWid=preImg.size[0]
	preHei=preImg.size[1]
	if preWid>preHei:
		tmp=preWid
		preWid=preHei
		preHei=tmp
	if preWid == 256:
		continue
	print picPath
	width=img.size[0]
	height=img.size[1]
	if width>height:
		newHeight=256
		newWidth=256*width/height
	else:
		newWidth=256
		newHeight=256*height/width
	try:
		newImg=img.resize((newWidth,newHeight),Image.ANTIALIAS)
		newImg.save(os.path.join(outputDir,file))
	except IOError:
		pass
log.close()
	#sys.exit()
