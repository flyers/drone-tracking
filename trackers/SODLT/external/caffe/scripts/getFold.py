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
#       newHeight=256
#       newWidth=256*width/height
#else:
#       newWidth=256
#       newHeight=256*height/width
#newImg=img.resize((newWidth,newHeight),Image.ANTIALIAS)
 #newImg.save("page.jpeg")
#newImg.save("/data/nwangab/n01440764/n01440764_1232.JPEG")

log=open('folds.log','w+')
#log.write("x")
#log.close()
dir='/data/kxmo/ImageNet/train_extract/'
folds=os.listdir(dir)
for fold in folds:
        print fold
        #continue
        #log=open('resize.log','w')
        log.write(fold+"\n")
        #log.close()
        #sys.exit()
        #files=os.listdir(os.path.join(dir,fold))
log.close()

