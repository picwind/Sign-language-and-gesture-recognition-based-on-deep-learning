import cv2
import os
import math


counter = 0
fps = 10
size = (512, 424)
path32 = "F:\\dxscxxm\\data\\image\\all32depth\\"
for ooo in range(7840):
    counter = counter + 1
    directory16 = math.ceil(counter/490)
    directory = str(directory16)
    s = (counter - (directory16 - 1)*490)/49
    fr = math.ceil(s)
    a = counter - (directory16-1)*490 - (fr-1)*49
    counter1 = str(a)
    counter2 = str(fr)
    new_name = "{:0>2}".format(counter2) + "{:0>2}".format(counter1)
    if (directory16 == 15) & (new_name == "0210"):
        filelist = os.listdir(path32 + directory + "\\" + new_name)
        filelist.sort(key=lambda x: int(x[:-4]))
        video = cv2.VideoWriter("F:\\dxscxxm\\data\\video\\all32video\\" + directory + "\\" + new_name + ".avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        for item in filelist:
            if item.endswith('.jpg'):
                 item = path32 + directory + "\\" + new_name + "\\" + item
                 img = cv2.imread(item)
                 video.write(img)
        video.release()
        cv2.destroyAllWindows()
        print(directory, new_name, "生成成功")
