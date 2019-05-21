from tkinter import *
from PIL import Image, ImageTk
import time
from GUI.widgets import *
import imageio
import imutils
import cv2
import test_one as ts
import numpy as np
import tensorflow as tf
import os


def action_index():  # 索引
    fr = open('./index.txt', 'r')
    List = list()
    for line in fr:
        v = line.strip('\\n').split()
        List += v
    fr.close()
    return List


def test_step(data, j):  # 测试
    y = ts.test(data)
    index = List[np.argmax(y)]
    tag.set(List[j - 1])
    res.set('……')
    top.update()
    video_path = data
    # video = imageio.get_reader(video_path, 'ffmpeg')
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while (True):
        ret, frame = cap.read()  # 捕获一帧图像
        if ret:
            videoFrame = imutils.resize(frame, width=512, height=424)
            image = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (170, 140))
            img = Image.fromarray(image)
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img)
            panel.image = img
            time.sleep(0.1)
            panel.update()
            idx = idx + 1
        else:
            break
    cap.release()
    res.set(index)
    top.update()
    return y


def start():  # 全部测试
    global STOP
    STOP = False
    cor, N = 0, 0
    for i in range(1, 4):
        for j in range(1, 50):
            if STOP == True:
                acc.set('0%')
                x.set('0%')
                change_schedule(0, 3*49)
                top.update()
                return
            else:
                N += 1
                data = './32video/4/{:0>2}{:0>2}.avi'.format(i, j)
                y = test_step(data, j)
                if np.argmax(y) == j - 1:
                    cor += 1
                acc.set('{:.2f}%'.format(cor * 100.0 / N))
                change_schedule(N, 3*49)
                top.update()

def one_test():  # 单项测试
    global STOP
    STOP = False
    num = lb.curselection()
    if num == ():
        num = (0, 1)
    tag.set(List[list(num)[0]])
    res.set('……')
    top.update()
    cor, N = 0, 0
    for i in range(1, 4):
        if STOP == True:
            acc.set('0%')
            x.set('0%')
            change_schedule(0, 3*49)
            top.update()
            return
        else:
            N += 1
            data = './32video/4/{:0>2}{:0>2}.avi'.format(i, list(num)[0] + 1)
            y = test_step(data, list(num)[0] + 1)
            if np.argmax(y) == list(num)[0]:
                cor += 1
            acc.set('{:.2f}%'.format(cor * 100.0 / N))
            change_schedule(N, 3)
            top.update()


def test_exit():  # 退出测试
    os._exit(0)


def stoptest():  # 停止测试
    global STOP
    STOP = True


def change_schedule(now_schedule, all_schedule):  # 更新进度条函数
    canvas.coords(fill_rec, (5, 5, 6 + (now_schedule / all_schedule) * 100, 25))
    top.update()
    x.set(str(round(now_schedule / all_schedule * 100, 2)) + '%')
    if round(now_schedule / all_schedule * 100, 2) == 100.00:
        x.set('完毕')


if __name__ == '__main__':
    # 索引
    List = action_index()

    # 初始化窗口
    top = Tk()
    top.title('基于深度学习的手势手语识别')
    top.geometry('540x360')

    # 初始化字符串流
    tag = StringVar()
    tag.set('聋奥手语')
    res = StringVar()
    res.set('……')
    acc = StringVar()
    acc.set('0%')
    startest = StringVar()
    startest.set('循环测试')
    x = StringVar()
    x.set('0%')
    STOP = False
    # 视频窗口
    load = Image.open('001.jpg')
    initIamge = ImageTk.PhotoImage(load, (170, 140))
    panel = Label(top, image=initIamge, textvariable=tag,
                  width=170, height=140, fg='white', font=('微软雅黑', 10, 'bold'),
                  compound='center')
    panel.image = initIamge
    panel.grid(row=0, column=0, columnspan=6, rowspan=6)

    # Labels
    # 识别结果
    Label(top, text='识别结果：', font=('微软雅黑', 10), width=8,
          fg='black', anchor=E).grid(row=6, column=0, rowspan=1)
    Label(top, textvariable=res, font=('微软雅黑', 10), width=8,
          anchor=E).grid(row=6, column=1, rowspan=1)
    # 正确率
    Label(top, text='正确率：', font=('微软雅黑', 10), width=8,
          fg='black', anchor=E).grid(row=9, column=0, rowspan=1)
    Label(top, textvariable=acc, font=('微软雅黑', 10), width=8,
          fg='black', anchor=E).grid(row=9, column=1, rowspan=1)
    # 测试进度
    Label(top, text='测试进度：', font=('微软雅黑', 10), width=8,
          fg='black').grid(row=10, column=0)

    Label(top, textvariable=x, font=('微软雅黑', 10), width=8,
          fg='black').grid(row=10, column=5)

    # Listbox
    lb = Listbox(top, width=12, selectmode=BROWSE)
    sl = Scrollbar(top)
    sl.grid(row=0, column=13, rowspan=4, ipady=60)
    lb['yscrollcommand'] = sl.set
    for item in List:
        lb.insert(END, item)
    lb.grid(row=0, column=12, rowspan=4)
    sl['command'] = lb.yview

    # Button
    Button(top, text='全部测试', font=('微软雅黑', 10), fg='black',
           command=start, width=10).grid(row=1, column=11, rowspan=2)

    Button(top, text='单项测试', font=('微软雅黑', 10), fg='black',
           command=one_test, width=10).grid(row=0, column=11, rowspan=2)

    Button(top, text='停止测试', font=('微软雅黑', 10), fg='black',
           command=stoptest, width=10).grid(row=2, column=11, rowspan=2)

    Button(top, text='退出', font=('微软雅黑', 10), fg='black',
           command=test_exit, width=10).grid(row=12, column=12)

    # 进度条
    # 创建画布
    canvas = Canvas(top, width=105, height=25)
    canvas.grid(row=10, column=1, columnspan=4)

    # 测试进度
    out_rec = canvas.create_rectangle(5, 5, 105, 25, outline='black', width=1)
    fill_rec = canvas.create_rectangle(5, 5, 5, 25, outline='', width=0, fill='green')

    top.mainloop()



