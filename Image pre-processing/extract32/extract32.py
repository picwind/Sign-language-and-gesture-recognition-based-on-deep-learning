import cv2
import os
import math


def loaddatadet(infile):
    f = open(infile, 'r')
    sourceinline = f.readlines()
    dataset = []
    for line in sourceinline:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        dataset.append(temp2)
    return dataset


infiles = "F:\\dxscxxm\\data\\image\\alldepth_jpg\\all.txt"
dataset1 = loaddatadet(infiles)
print(dataset1)
# for i in range(16):
#     readpath = "F:\\dxscxxm\\data\\video\\rawvideo\\depth\\" + str(i + 1)
#     avi_name = os.listdir(readpath)
#     o = 0
#     for p in avi_name:
#         o = o + 1
#         s = o / 49
#         f = math.ceil(s)
#         a = o - (f - 1) * 49
#         conter1 = str(a)
#         conter2 = str(f)
#         new_name = "{:0>2}".format(conter2) + "{:0>2}".format(conter1)
#         real_read_path = "F:\\dxscxxm\\data\\video\\rawvideo\\depth\\" + str(i + 1) + "\\" + p
#         if ~os.path.exists("F:\\dxscxxm\\data\\image\\all32depth\\" + str(i + 1) + "\\" + new_name):
#             os.makedirs("F:\\dxscxxm\\data\\image\\all32depth\\" + str(i + 1) + "\\" + new_name)
counter = 0
path32 = "F:\\dxscxxm\\data\\image\\all32depth\\"
for start_end_frame in dataset1:
    counter = counter + 1
    directory16 = math.ceil(counter/490)
    directory = str(directory16)
    s = (counter - (directory16 -1)*490)/49
    fr = math.ceil(s)
    a = counter - (directory16-1)*490 - (fr-1)*49
    counter1 = str(a)
    counter2 = str(fr)
    new_name = "{:0>2}".format(counter2) + "{:0>2}".format(counter1)
    if (directory16 == 16) & (new_name == "0132"):
        begin = int(start_end_frame[0])
        end = int(start_end_frame[1])
        lenghth = end - begin + 1
        print(new_name, begin, end)
        for ss in range(32):
            oss = ss + 1
            alpha = 1 - (lenghth / 32) * oss + math.floor((lenghth / 32) * oss)
            beta = 1 - alpha
            left = begin + math.floor((lenghth / 32) * oss) - 1
            right = left + 1
            if left < begin:
                left = begin
                right = begin
            if right > end:
                right = end
                left = end
            lpic = cv2.imread(
                "F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + directory + "\\" + new_name + "\\" + str(left) + ".jpg")
            rpic = cv2.imread(
                "F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + directory + "\\" + new_name + "\\" + str(right) + ".jpg")
            dst = cv2.addWeighted(lpic, alpha, rpic, beta, 0)
            cv2.imwrite(path32 + directory + "\\" + new_name + "\\" + str(oss) + ".jpg", dst)
        print("生成成功")
    if os.path.exists(path32 + directory + "\\" + new_name):
        print("已存在")
    else:
        os.makedirs(path32 + directory + "\\" + new_name)
        begin = int(start_end_frame[0])
        end = int(start_end_frame[1])
        lenghth = end - begin + 1
        print(new_name, begin, end)
        for standardf in range(32):

           stf = standardf + 1
           alpha = 1 - (lenghth/32)*stf + math.floor((lenghth/32)*stf)
           beta = 1 - alpha
           left = begin + math.floor((lenghth/32)*stf) - 1
           right = left + 1
           if left < begin:
               left = begin
           if right > end:
               right = end
           lpic = cv2.imread("F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + directory + "\\" + new_name + "\\" + str(left) + ".jpg")
           rpic = cv2.imread("F:\\dxscxxm\\data\\image\\alldepth_jpg\\" + directory + "\\" + new_name + "\\" + str(right) + ".jpg")
           dst = cv2.addWeighted(lpic, alpha, rpic, beta, 0)
           cv2.imwrite(path32 + directory + "\\" + new_name + "\\" + str(stf) + ".jpg", dst)
        print("生成成功")
