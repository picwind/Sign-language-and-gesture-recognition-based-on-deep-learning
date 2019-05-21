#!/usr/bin/ env python -*- coding:UTF-8 -*-
import cv2
import os
import math
for i in range(16):
    readpath = "F:\\dxscxxm\\data\\video\\rawvideo\\depth\\" + str(i+1)
    avi_name = os.listdir(readpath)
    o = 0
    for p in avi_name:
        o = o + 1
        s = o/49
        f = math.ceil(s)
        a = o - (f - 1) * 49
        conter1 = str(a)
        conter2 = str(f)
        new_name = "{:0>2}".format(conter2) + "{:0>2}".format(conter1)
        real_read_path = "F:\\dxscxxm\\data\\video\\rawvideo\\depth\\" + str(i+1) + "\\" + p
        vc = cv2.VideoCapture(real_read_path)
        os.makedirs("F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + str(i+1) + "\\" + new_name)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        timeF = 1
        while rval:
            rval, frame = vc.read()
            pathout = "F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + str(i+1) + "\\" + new_name + "\\"
            if (c%timeF==0):
                cv2.imwrite(pathout + str(c) + ".jpg", frame)
            c = c + 1
            cv2.waitKey(1)
        vc.release()
