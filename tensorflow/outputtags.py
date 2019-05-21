import math
import random


def outputtraintag(num, filename, format1,):
    a_list = range(1, 17)
    # b_list = random.sample(a_list, num)
    b_list = [4, 5, 9, 10]
    retu = []
    for element in range(16):
        if (element + 1) not in b_list:
            retu.append((element+1))
    print(retu)
    taglist = []
    for i in retu:
        o = 0
        for p in range(490):
            o = o + 1
            s = o / 49
            f = math.ceil(s)
            a = o - (f - 1) * 49
            conter1 = str(a)
            conter2 = str(f)
            new_name = "{:0>2}".format(conter2) + "{:0>2}".format(conter1)
            pathname = "/media/liuyu/_data2/大创/" + filename + "/" + str(i) + "/" + new_name + "." + format1 + " " + str(a-1) + "\n"
            taglist.append(pathname)
    with open("D:\\pythonworkspace\\hanjia\\" + filename + "train" + ".txt", "w")as f1:
        for path in taglist:
            f1.write(path)
        f1.close()
        print("生成标签文件成功")
    return b_list


def outputtesttag(n, filename, format1):
    taglist = []
    for ne in n:
        o = 0
        for p in range(490):
            o = o + 1
            s = o / 49
            f = math.ceil(s)
            a = o - (f - 1) * 49
            conter1 = str(a)
            conter2 = str(f)
            new_name = "{:0>2}".format(conter2) + "{:0>2}".format(conter1)
            pathname = "/media/liuyu/_data2/大创/" + filename + "/" + str(ne) + "/" + new_name + "." + format1 + " " + str(a-1) + "\n"
            taglist.append(pathname)
    with open("D:\\pythonworkspace\\hanjia\\" + filename + "test" + ".txt", "w")as ff:
        for path in taglist:
            ff.write(path)
        ff.close()
        print("生成标签文件成功")


# outputtraintag(12, "allRF", "jpg")
# outputtesttag(13, "allRF", "jpg")
n = outputtraintag(12, "allFR", "jpg")
outputtesttag(n, "allFR", "jpg")
# n = outputtraintag(12, "all32video", "avi")
# outputtesttag(n, "all32video", "avi")
