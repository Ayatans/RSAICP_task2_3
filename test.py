import glob
import math
from math import sqrt, atan, floor
import utils.datasets as ds
import os
from pathlib import Path
import shutil
from utils.general import kmean_anchors
from PIL import Image
from PIL import ImageDraw
import random
import time
import numpy as np
import json
import cv2

# 生成图片文件名的txt
# img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
#
#
#
# out='E:/Codes/RotatedYOLOv5/detection'
# filenamepath = out + os.sep + 'result_txt' + os.sep + 'imgnamefile.txt'
# val=r'E:\Papers\dataset\split\img'
# with open(filenamepath, 'w') as filename:
#     for fs in os.listdir(val):
#         fs_name = os.path.splitext(fs)
#         fs_destfile = fs_name[0]
#         filename.write(fs_destfile)
#         filename.write('\n')


#
# # kmeans计算候选框尺寸

# c=kmean_anchors(path='./data/rsaicp.yaml', n=9, img_size=512, thr=4.0, gen=1000, verbose=True)
# print(c)

# # # 计算标签框的长段边及其比例
# path=r'E:\Papers\dataset\train'
# maskpath=path+os.sep+'mask'
# longside=[]
# shortside=[]
# for root, dirs, files in os.walk(maskpath):
#     for file in files:
#         filepath=maskpath+os.sep+file
#         with open(filepath,'r') as f:
#             lines=f.readlines()
#             for line in lines:
#                 x=line.split(',')
#                 x1=float(x[1])
#                 y1=float(x[2])
#                 x2=float(x[3])
#                 y2=float(x[4])
#                 x3=float(x[5])
#                 y3=float(x[6])
#                 # x=line.split(' ')
#                 # x1=float(x[0])
#                 # y1=float(x[1])
#                 # x2=float(x[2])
#                 # y2=float(x[3])
#                 # x3=float(x[4])
#                 # y3=float(x[5])
#                 side1=sqrt(abs(x2-x3)**2+abs(y2-y3)**2)
#                 side2=sqrt(abs(x1-x2)**2+abs(y2-y1)**2)
#                 longside.append(max(side1,side2))
#                 shortside.append(min(side1,side2))
#
# print(longside)
# print(shortside)
# ratio=[]
# for i in range(len(longside)):
#     ratio.append(longside[i]/shortside[i])
# ratio.sort()
# print(ratio)
#
# longside.sort()
# shortside.sort()
# print(longside)
# print(shortside)


# # # 计算旋转框角度
# path=r'E:\Papers\dataset\train'
# maskpath=path+os.sep+'mask'
# theta=[]
# for root, dirs, files in os.walk(maskpath):
#     for file in files:
#         filepath=maskpath+os.sep+file
#         with open(filepath,'r') as f:
#             lines=f.readlines()
#             for line in lines:
#                 x=line.split(',')
#                 x1=float(x[1])
#                 y1=float(x[2])
#                 x2=float(x[3])
#                 y2=float(x[4])
#                 x3=float(x[5])
#                 y3=float(x[6])
#                 # x=line.split(' ')
#                 # x1=float(x[0])
#                 # y1=float(x[1])
#                 # x2=float(x[2])
#                 # y2=float(x[3])
#                 # x3=float(x[4])
#                 # y3=float(x[5])
#                 the=atan(abs(x2-x1)/abs(y2-y1))   # 返回弧度值
#                 theta.append(the)
# print(theta)
# theta.sort()
# print(theta)
# print(len(theta))
#
# # # 旋转图片和标签
# imgpath=r'E:\Papers\dataset\split512-0.2-150-ch3-r\img/'
# maskpath=r'E:\Papers\dataset\split512-0.2-150-ch3-r\mask/'
# dst90path=r'E:\Papers\dataset\split512-0.2-150-ch3-90'
# dst180path=r'E:\Papers\dataset\split512-0.2-150-ch3-180'
# dst270path=r'E:\Papers\dataset\split512-0.2-150-ch3-270'
#
#
# imglist=os.listdir(imgpath)
# masklist=os.listdir(maskpath)
# for img in imglist:
#     image=Image.open(imgpath+img)
#     size=image.size[0]
#     name = img.split('.')[0]
#     for i in range(1,4):
#         dstpath=eval('dst'+str(i*90)+'path')
#         tmppath = dstpath +os.sep +'img/' + name +'-'+str(i*90)+'.png'
#         lpath=dstpath+os.sep+'mask'
#         if not os.path.exists(dstpath):
#             os.mkdir(dstpath)
#         if not os.path.exists(dstpath+os.sep+'img/'):
#             os.mkdir(dstpath+os.sep+'img/')
#         if not os.path.exists(lpath):
#             os.mkdir(lpath)
#
#
#         image.rotate(90*i).save(tmppath)    # 逆时针
#         label=maskpath+name+'.txt'
#         with open(label, 'r') as f:
#             lines=f.readlines()
#         for line in lines:
#             res=line.split(' ')
#             x1=float(res[0])
#             y1=float(res[1])
#             x2=float(res[2])
#             y2=float(res[3])
#             x3=float(res[4])
#             y3=float(res[5])
#             x4=float(res[6])
#             y4=float(res[7])
#             if i==1:
#                 t=x1
#                 x1=y1
#                 y1=size-t
#                 t=x2
#                 x2=y2
#                 y2=size-t
#                 t=x3
#                 x3=y3
#                 y3=size-t
#                 t=x4
#                 x4=y4
#                 y4=size-t
#                 newx1=x4
#                 newy1=y4
#                 newx2=x1
#                 newy2=y1
#                 newx3=x2
#                 newy3=y2
#                 newx4=x3
#                 newy4=y3
#                 new=str(newx1)+' '+str(newy1)+' '+str(newx2)+' '+str(newy2)+' '+str(newx3)+' '+str(newy3)+' '+str(newx4)+' '+str(newy4)+' '+'bigship'+'\n'
#             if i==2:
#                 x1=size-x1
#                 y1=size-y1
#                 x2=size-x2
#                 y2=size-y2
#                 x3=size-x3
#                 y3=size-y3
#                 x4=size-x4
#                 y4=size-y4
#                 newx1=x3
#                 newy1=y3
#                 newx2=x4
#                 newy2=y4
#                 newx3=x1
#                 newy3=y1
#                 newx4=x2
#                 newy4=y2
#                 new=str(newx1)+' '+str(newy1)+' '+str(newx2)+' '+str(newy2)+' '+str(newx3)+' '+str(newy3)+' '+str(newx4)+' '+str(newy4)+' '+'bigship'+'\n'
#             if i==3:
#                 t=x1
#                 x1=size-y1
#                 y1=t
#                 t=x2
#                 x2=size-y2
#                 y2=t
#                 t=x3
#                 x3=size-y3
#                 y3=t
#                 t=x4
#                 x4=size-y4
#                 y4=t
#                 newx1=x2
#                 newy1=y2
#                 newx2=x3
#                 newy2=y3
#                 newx3=x4
#                 newy3=y4
#                 newx4=x1
#                 newy4=y1
#                 new=str(newx1)+' '+str(newy1)+' '+str(newx2)+' '+str(newy2)+' '+str(newx3)+' '+str(newy3)+' '+str(newx4)+' '+str(newy4)+' '+'bigship'+'\n'
#             with open(lpath+os.sep+name+'-'+str(i*90)+'.txt', 'a') as f:
#                 f.write(new)


# # # # 去除文件名中的副本二字
# path=r'E:\Papers\dataset\split512-0.2-rlight\yolo_labels'
# imglist=os.listdir(path)
# newlist=[]
# for imgname in imglist:
#     if '副本' in imgname:
#         imgpath = path + os.sep + imgname
#         newname=imgname.split('.')
#         ext=newname[1]
#         originalname=newname[0]
#         new=imgname.replace(' - 副本','_copy')
#         print(new)
#         os.rename(imgpath,path+os.sep+new)
#


# # 为了yolov5原版，去除标签最后一项角度
# path=r'E:\Papers\dataset\split512-0.2-r-val\labels'
# labellist=os.listdir(path)
# for label in labellist:
#     finalpath=path+os.sep+label
#     with open(finalpath, 'r') as f:
#         lines=f.readlines()
#         newlines=[]
#         for line in lines:
#             res=line.split(' ')
#             res=res[:-1]
#             res=str(res[0])+' '+str(res[1])+' '+str(res[2])+' '+str(res[3])+' '+str(res[4])
#             newlines.append(res)
#     file = open(finalpath, 'w').close()
#     with open(finalpath, 'a') as f:
#         for line in newlines:
#             f.write(line+'\n')


# # # # # 从all数据集中抽取一些空的图片放入训练集
# 

# def random_int_list(start, stop, length):
#     start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
#     length = int(abs(length)) if length else 0
#     random_list = []
#     for i in range(length):
#         random_list.append(random.randint(start, stop))
#     return random_list
# 
# allpath='E:\Papers\dataset\split512-0.2all'
# allimgpath=allpath+os.sep+'img'
# alllabelpath=allpath+os.sep+'mask'
# imglist=os.listdir(allimgpath)
# dstpath=r'E:\Papers\dataset\11'
# randind=random_int_list(1, 64851, 1000)
# for img in imglist:
#     ind=imglist.index(img)
#     if ind not in randind:
#         continue
#     name=img.split('.')[0]
#     empty=True
#     mask=alllabelpath+os.sep+name+'.txt'
#     with open(mask, 'r') as f:
#         if f.readlines():
#             empty=False
#     if not empty:
#         continue
#     shutil.copyfile(allimgpath+os.sep+img, dstpath+os.sep+'img'+os.sep+img)
#     shutil.copyfile(mask, dstpath+os.sep+'mask'+os.sep+name+'.txt')

#
# ### 统计图片各点的灰度值（用来排除陆地）
# path=r'E:\Papers\dataset\train\img'
# boxsize=500
# thresh=boxsize*boxsize/2
# bright_thresh=175
# for i in range(1,26):
#     imgpath=path+os.sep+str(i)+'.png'
#     img=Image.open(imgpath)
#     imgarr=np.array(img)
#     mean_light=np.mean(imgarr)
#     standard_var_light=sqrt(np.var(imgarr))
#     print('img%d: mean %f var: %f'% (i, mean_light, standard_var_light))
#     a=ImageDraw.ImageDraw(img)
#     width,height=img.size
#     for x in range(1,width-boxsize,boxsize):
#         for y in range(1,height-boxsize,boxsize):
#             box=(x,y,x+boxsize,y+boxsize)
#             region=img.crop(box)
#             regionarr=np.array(region)
#             if np.sum(regionarr>mean_light+standard_var_light)>thresh:
#                 # print(box)
#                 # region.show()
#                 a.rectangle(((box[0],box[1]),(box[2], box[3])), fill=None, outline='black', width=20)
#     img.save('land_detection/land'+str(i)+'.png')
# # box=(7500,1500,8000,2000)   # 设定裁剪位置，左上角点wh和右下角点wh
# # # box=(8000,6000,8500,6500)   # 设定裁剪位置，左上角点wh和右下角点wh
# # region=img.crop(box)
# # region.show()
# # regionarr=np.array(region)
# # thresharr=regionarr>150
# # print(thresharr)
# # n=np.sum(thresharr)
# # print(n)

# ### 统计某个数据集共有多少个目标
# path=r'E:\Papers\dataset\split512-0.2'
# imgpath=os.path.join(path, 'img')
# maskpath=os.path.join(path, 'mask')
# masklist=os.listdir(maskpath)
# n=0
# for i in masklist:
#     absloc=os.path.join(maskpath, i)
#     with open(absloc, 'r') as f:
#         lines=f.readlines()
#         n+=len(lines)
#
# print(n)


# # # # 将删掉的图片的标签同时删掉
# imgpath=r'E:\Papers\dataset\split512-0.2-150-ch3\img'
# maskpath=r'E:\Papers\dataset\split512-0.2-150-ch3\mask'
# imglist=os.listdir(imgpath)
# masklist=os.listdir(maskpath)
# for mask in masklist:
#     name=mask.split('.')[0]
#     imgname=name+'.png'
#     if imgname not in imglist:
#         deletepath=maskpath+os.sep+mask
#         os.remove(deletepath)


# # # 获取val集
# valpath=r'E:\datasets\832\valfixed-r15n-832'
# trainpath=r'E:\datasets\832\allfixed-r15n-832'
# prefix=['1','3','10','12','13','19','20','23']
# imglist=os.listdir(trainpath+os.sep+'img')
# masklist=os.listdir(trainpath+os.sep+'mask')
# # yololist=os.listdir(trainpath+os.sep+'yolo_labels')
# for i in imglist:
#     name=i.split('.')[0]
#     pre=name.split('__')[0]
#     if '-' in pre:
#         pre=pre.split('-')[0]
#     if pre in prefix:
#         moveimg=trainpath+os.sep+'img'+os.sep+i
#         movemask=trainpath+os.sep+'mask'+os.sep+name+'.txt'
#         # moveyolo=trainpath+os.sep+'yolo_labels'+os.sep+name+'.txt'
#         shutil.copy(moveimg, valpath+'/img/')
#         shutil.copy(movemask, valpath+'/mask/')
#         # shutil.copy(moveyolo, valpath+'/yolo_labels')
#         #os.remove(moveimg)
#         #os.remove(movemask)
#         # os.remove(moveyolo)


# # 将kaggle数据集的json标签转为mask
# jsonpath=r'E:\Codes\instances_ships_train2018.json'
# txtpath=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection/txt'
# with open(jsonpath, 'r') as labeljson:
#     label = json.load(labeljson)
# labellist=label["annotations"]   # 列表的每个元素是一个字典，代表一张图的标注
# images=label["images"]
# imgid={}
# for i in images:
#     id=i["id"]
#     imgid[id]=i["file_name"]
# for dic in labellist:
#     id=dic["image_id"]
#     name=imgid[id].split('.')[0]
#     txtname=name+'.txt'
#     writepath=os.path.join(txtpath, txtname)
#     bbox=dic["bbox"] # list:x,y,w,h
#     x=bbox[0]
#     y=bbox[1]
#     w=bbox[2]
#     h=bbox[3]
#     x1=x
#     x2=x1
#     x3=x+w
#     x4=x3
#     y1=y
#     y2=y+h
#     y3=y2
#     y4=y1
#     line='bigship,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+'\n'
#     with open(writepath, 'a') as f:
#         f.write(line)


# # 将kaggle数据集的json标签转为旋转框的mask
# jsonpath=r'E:\Codes\instances_ships_train2018.json'
# txtpath=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection/rtxt'
# with open(jsonpath, 'r') as labeljson:
#     label = json.load(labeljson)
# labellist=label["annotations"]   # 列表的每个元素是一个字典，代表一张图的标注
# images=label["images"]
# imgid={}
# m=0
# n=0
# for i in images:
#     id=i["id"]
#     imgid[id]=i["file_name"]
# for dic in labellist:
#     id=dic["image_id"]
#     name=imgid[id].split('.')[0]
#     txtname=name+'.txt'
#     writepath=os.path.join(txtpath, txtname)
#     bbox=dic["segmentation"] # list:x,y,w,h
#     if len(bbox)>1:
#         print(name)
#         n+=1
#         continue
#     bbox = bbox[0]
#     if len(bbox)>10:
#         print('more than 10:', name)
#         m+=1
#         continue
#     if len(bbox)==10:
#         x1=bbox[0]
#         y1=bbox[1]
#         x2=bbox[2]
#         y2=bbox[3]
#         x3=bbox[4]
#         y3=bbox[5]
#         x4=bbox[6]
#         y4=bbox[7]
#         line='bigship,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+'\n'
#         with open(writepath, 'a') as f:
#             f.write(line)
# print('more than 1 list:',n,'\nmore than 10 num:',m)

# ### 将kaggle中不存在图片的标签去掉
# path=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection'
# train=path+'/all'
# txt=path+'/txt'
# imglist=os.listdir(train)
# for i in os.listdir(txt):
#     name=i.split('.')[0]
#     imgname=name+'.jpg'
#     if imgname not in imglist:
#         os.remove(os.path.join(txt, i))


# # # # # 筛选出kaggle数据集中尺寸过大或过小的图
# path=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection'
# savepath=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\selected'
# allship=os.path.join(path,'train')
# mask=os.path.join(path, 'txt')
# imglist=os.listdir(allship)
# selected=[]
# for img in imglist:
#     name=img.split('.')[0]
#     txtname=name+'.txt'
#     txtpath=os.path.join(mask, txtname)
#     del_flag=False
#     with open(txtpath, 'r') as f:
#         lines=f.readlines()
#         for line in lines:
#             part=line.split(',')
#             x1=part[1]
#             y1=part[2]
#             x2=x1
#             y2=part[4]
#             x3=part[5]
#             y3=y2
#             x4=x3
#             y4=y1
#             side1=abs(float(x3)-float(x1))
#             side2=abs(float(y2)-float(y1))
#             if (side1<30 and side2 <30):# or (side1>300 and side2>300):
#                 selected.append(img)
# for i in selected:
#     shutil.copy(allship+os.sep+i, savepath)


# # # # 将四个角度旋转过的训练集打乱
#
# path=r'E:\Papers\dataset\splitfixed512-0.2-r-train'
# imgpath=os.path.join(path, 'img')
# saveimgpath=os.path.join(path, 'randimg')
# maskpath=os.path.join(path, 'mask')
# savemaskpath=os.path.join(path, 'randmask')
# yolopath=os.path.join(path, 'yolo_labels')
# saveyolopath=os.path.join(path, 'randyolo_labels')
# # labelspath=os.path.join(path, 'labels')
# # savelabelspath=os.path.join(path, 'randlabels')
# imglist=os.listdir(imgpath)
# for img in imglist:
#     name=img.split('.')[0]
#     prenum=random.randint(1,100)
#     newname=str(prenum)+'_'+name
#     os.rename(imgpath+'/'+img, imgpath+'/'+newname+'.png')
#     os.rename(maskpath+'/'+name+'.txt', maskpath+'/'+newname+'.txt')
#     os.rename(yolopath+'/'+name+'.txt', yolopath+'/'+newname+'.txt')
#     # os.rename(labelspath+'/'+name+'.txt', labelspath+'/'+newname+'.txt')


# # # # 将rtxt中的，没有图的部分删掉
# rtxt=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\rtxt'
# train=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\train'
# copypath=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\rtxtimg'
# imglist=os.listdir(train)
# txtlist=os.listdir(rtxt)
# for i in txtlist:
#     name=i.split('.')[0]
#     imgname=name+'.jpg'
#     imgpath=os.path.join(train, imgname)
#     shutil.copy(imgpath, copypath)


# # # 将筛选后的kaggle的标签和yolo标签同时筛选出来
# imgpath=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\selected'
# dst=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\label_for_selected'
# src=r'F:\BaiduNetdiskDownload\kaggle-airbus-ship-detection\yolo_labels'
# imglist=os.listdir(imgpath)
# for i in imglist:
#     name=i.split('.')[0]
#     txtname=name+'.txt'
#     txtsrc=os.path.join(src, txtname)
#     shutil.copy(txtsrc, dst)


# # # # 将数据集打乱
# imgpath=r'E:\Papers\dataset\kaggle_trainfixed\img'
# yolopath=r'E:\Papers\dataset\kaggle_trainfixed/yolo_labels'
# imglist=os.listdir(imgpath)
# for img in imglist:
#     name=img.split('.')[0]
#     prenum=random.randint(1,100)
#     newname=str(prenum)+'_'+name
#     os.rename(imgpath+'/'+img, imgpath+'/'+newname+'.png')
#     os.rename(yolopath+'/'+name+'.txt', yolopath+'/'+newname+'.txt')


# # # yolo展示box
# imgpath=r'E:\Papers\dataset\splitfixed512-0.2-r-rand-train\img'
# maskpath=r'E:\Papers\dataset\splitfixed512-0.2-r-rand-train\mask'
# yolopath=r'E:\Papers\dataset\splitfixed512-0.2-r-rand-train\yolo_labels'
# txtname='33_6__1__7240___5792-270.txt'
# imgname='33_6__1__7240___5792-270.png'
# img = Image.open(imgpath+os.sep+imgname)
# a = ImageDraw.ImageDraw(img)
# with open(yolopath+os.sep+txtname, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         item = line.split(' ')
#         x=float(item[0])*512
#         y=float(item[1])*512
#
#         w=float(item[2])*512
#         h=float(item[3])*512
#         angle=float(item[4])
#         wx=w*math.cos(angle)
#         wy=w*math.sin(angle)
#         hx=h*math.cos(angle)
#         hy=h*math.sin(angle)
#         x1=x
#         y1=y
#         x2=x1+hx
#         y2=x1+hy
#         x3=x2+wx
#         y3=y2-wy
#         x4=x3-hx
#         y4=y3-hy
#         a.point([x, y])
#         a.point([x2, y2])
#         a.point([x3, y3])
#         a.point([x4, y4])
#         # a.line([(x1,y1), (x2,y2)], width=5)
#         # a.line([(x2,y2)   , (x3,y3)], width=5)
#         # a.line([(x3,y3), (x4,y4)], width=5)
#         # a.line([(x4,y4), (x1,y1)], width=5)
#
# img.save(imgname)

#
# # # # 将文件名写入txt
# path=r'E:\datasets\832\allfixed-r90n-832-1-300'
# src=r'E:\datasets\832\allfixed-r90n-832-1-300'
# train=os.listdir(src+'/img')
# namelist=[]
# for i in train:
#     namelist.append(i.split('.')[0])
# lines=''
# for i in namelist:
#     lines+=i+'\n'
# with open(path+'/train.txt', 'w') as f:
#
#     f.write(lines)


# # # 获取旋转
# def Srotate(angle,valuex,valuey,pointx,pointy):
#   valuex = np.array(valuex)
#   valuey = np.array(valuey)
#   sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
#   sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
#   return sRotatex,sRotatey
#
#
# path=r'E:\datasets\original'
# imgpath=path+'/img/'
# maskpath=path+'/fixed/'
# imglist=os.listdir(imgpath)
# for img in imglist:
#     image = Image.open(imgpath + img)
#     name=img.split('.')[0]
#     txtname=name+'.txt'
#     xc=image.size[0]/2
#     yc=image.size[1]/2
#     for i in range(1,24):
#         dstpath='E:\datasets/fixedrotate/train-r'+str(i*15)
#         tmppath = dstpath +os.sep +'img/' + name +'-'+str(i*15)+'.png'
#         lpath=dstpath+os.sep+'mask'
#         if not os.path.exists(dstpath):
#             os.mkdir(dstpath)
#         if not os.path.exists(dstpath+os.sep+'img/'):
#             os.mkdir(dstpath+os.sep+'img/')
#         if not os.path.exists(lpath):
#             os.mkdir(lpath)
#
#
#         image.rotate(15*i).save(tmppath)    # 逆时针
#         label=maskpath+txtname
#         with open(label, 'r') as f:
#             lines=f.readlines()
#             for line in lines:
#                 res=line.split(',')
#                 x1=float(res[1])
#                 y1=float(res[2])
#                 x2=float(res[3])
#                 y2=float(res[4])
#                 x3=float(res[5])
#                 y3=float(res[6])
#                 x4=float(res[7])
#                 y4=float(res[8])
#                 newx1, newy1=Srotate(i*math.pi/12, x1, y1, xc, yc)
#                 newx2, newy2=Srotate(i*math.pi/12, x2, y2, xc, yc)
#                 newx3, newy3=Srotate(i*math.pi/12, x3, y3, xc, yc)
#                 newx4, newy4=Srotate(i*math.pi/12, x4, y4, xc, yc)
#
#                 new='bigship'+','+str(newx1)+','+str(newy1)+','+str(newx2)+','+str(newy2)+','+str(newx3)+','+str(newy3)+','+str(newx4)+','+str(newy4)+'\n'
#                 with open(lpath+os.sep+name+'-'+str(i*15)+'.txt', 'a') as f:
#                     f.write(new)
#     print(name)


# 删除空标签图片
# for h in range(1,24):
#     trainpath=r'E:\datasets\832/trainfixed-r'+str(h*15)+'-split832-5-300'
#     imglist=os.listdir(trainpath+'/img/')
#     masklist=os.listdir(trainpath+'/mask/')
#     for i in masklist:
#         name=i.split('.')[0]
#         pngname=name+'.png'
#         size=os.path.getsize(trainpath+'/mask/'+i)
#         if not size:
#             if os.path.exists(trainpath+'/img/'+pngname):
#                 os.remove(trainpath+'/img/'+pngname)
#             os.remove(trainpath+'/mask/'+i)
#     print(h*15)

#
# trainpath=r'E:\datasets\splitfixed832-0.8-300'
# imglist=os.listdir(trainpath+'/img/')
# masklist=os.listdir(trainpath+'/mask/')
# for i in imglist:
#     name=i.split('.')[0]
#     txtname=name+'.txt'
#     size=os.path.getsize(trainpath+'/mask/'+txtname)
#     if not size:
#         os.remove(trainpath+'/img/'+i)
#         os.remove(trainpath+'/mask/'+txtname)


# # # 生成多尺度图像
# path=r'E:\datasets\832\all-r90n-832v2'
# imgpath=path+'/img/'
# maskpath=path+'/mask/'
# imglist=os.listdir(imgpath)
# ratio=0.5
# for img in imglist:
#     image = Image.open(imgpath + img)
#     name=img.split('.')[0]
#     txtname=name+'.txt'
#     imgsize=image.size[0]
#     newsize = round(imgsize*ratio)
#     xc=image.size[0]/2
#     yc=image.size[1]/2
#     dstpath=r'E:\datasets\832\resize'+str(ratio)
#     tmppath = r'E:\datasets\832/resize'+str(ratio) +'/img/' + name +'-x'+str(ratio).replace('.', '')+'.png'
#     lpath=dstpath+os.sep+'mask'
#     if not os.path.exists(dstpath):
#         os.mkdir(dstpath)
#     if not os.path.exists(dstpath+os.sep+'img/'):
#         os.mkdir(dstpath+os.sep+'img/')
#     if not os.path.exists(lpath):
#         os.mkdir(lpath)
#
#
#     image.resize((newsize, newsize),Image.ANTIALIAS).save(tmppath)
#     label=maskpath+txtname
#     with open(label, 'r') as f:
#         lines=f.readlines()
#         for line in lines:
#             res=line.split(' ')
#             x1=float(res[0])
#             y1=float(res[1])
#             x2=float(res[2])
#             y2=float(res[3])
#             x3=float(res[4])
#             y3=float(res[5])
#             x4=float(res[6])
#             y4=float(res[7])
#             resize_ratio=newsize/imgsize
#             newx1,newy1,newx2,newy2,newx3,newy3,newx4,newy4=round(x1*resize_ratio)*1.0,round(y1*resize_ratio)*1.0,round(x2*resize_ratio)*1.0,round(y2*resize_ratio)*1.0,round(x3*resize_ratio)*1.0,round(y3*resize_ratio)*1.0,round(x4*resize_ratio)*1.0,round(y4*resize_ratio)*1.0,
#             new=str(newx1)+' '+str(newy1)+' '+str(newx2)+' '+str(newy2)+' '+str(newx3)+' '+str(newy3)+' '+str(newx4)+' '+str(newy4)+' '+'bigship'+'\n'
#             with open(lpath+os.sep+name+'-x'+str(ratio).replace('.', '')+'.txt', 'a') as f:
#                 f.write(new)


# # # # 取出旋转图像中的90度旋转图像
# path = r'E:\datasets\832\allfixed-r15n-832-1-300'
# dstpath = r'E:\datasets\832\allfixed-r90n-832-1-300'
# dstimgpath = os.path.join(dstpath, 'img')
# dstmaskpath = os.path.join(dstpath, 'mask')
# imgpath = os.path.join(path, 'img')
# maskpath = os.path.join(path, 'mask')
# imglist = os.listdir(imgpath)
# for i in imglist:
#     name = os.path.splitext(i)[0]
#     txtname = name + '.txt'
#     if not '-' in name:
#         shutil.copy(os.path.join(imgpath, i), dstimgpath)
#         shutil.copy(os.path.join(maskpath, txtname), dstmaskpath)
#     else:
#         degree = name.split('__')[0].split('-')[1]
#         if int(degree) % 90 == 0:
#             shutil.copy(os.path.join(imgpath, i), dstimgpath)
#             shutil.copy(os.path.join(maskpath, txtname), dstmaskpath)


# # # # 将boxshow中不存在的图像删除原图（refine）
# showpath=r'E:\Codes\RotatedYOLOv5\results'
# showlist=os.listdir(showpath)
# srcpath=r'E:\datasets\832\all-r90n-832v2'
# imgpath=os.path.join(srcpath, 'img')
# maskpath=os.path.join(srcpath, 'mask')
# n=0
# srcimglist=os.listdir(imgpath)
# for i in srcimglist:
#     name=os.path.splitext(i)[0]
#     showname='results'+i
#     if showname not in showlist:
#         os.remove(os.path.join(imgpath, i))
#         os.remove(os.path.join(maskpath, name+'.txt'))
#         print(showname)
#         n+=1
# print(n)


# # # 去除过亮的点，不好应用在旋转框上。
# obj = []
# result_path = '../output_path/ship_results.json'
# filepath = []
# with open(result_path, 'r') as f:
#     result0=json.load(f)
# new_result=[]
# for i in result0:  # 遍历每个图的结果
#     resulti = {}
#     imgname=i['image_name']
#     imgpath='../input_path/img/'+imgname
#     img = Image.open(imgpath)
#     imgarr = np.array(img)
#     mean_light = np.mean(imgarr)
#     standard_var_light = sqrt(np.var(imgarr))
#     labels=i['labels']
#     print(imgname)
#     remove_index=[]
#     for index, j in enumerate(labels): # 遍历一幅图里所有坐标结果，一个j代表一个框
#         points=j['points']
#         x1, y1=points[0][0],points[0][1]
#         x2, y2=points[1][0],points[1][1]
#         x3, y3=points[2][0],points[2][1]
#         x4, y4=points[3][0],points[3][1]
#         newpoint=[]
#         regionsize = sqrt((abs(x1 - x2)) ** 2 + (abs(y1 - y2)) ** 2) * sqrt((abs(x1 - x4)) ** 2 + (abs(y1 - y4)) ** 2)
#         regionsize = floor(regionsize)
#         print('regionsize=', regionsize)
#         num_thresh = round(regionsize / 2)
#         print('num_thresh=', num_thresh)
#         box = (min(int(x1),int(x2),int(x3),int(x4)), min(int(y1),int(y2),int(y3),int(y4)), max(int(x1),int(x2),int(x3),int(x4)), max(int(y1),int(y2),int(y3),int(y4)))
#         print('box=', box)
#         region = img.crop(box)
#         regionarr = np.array(region)
#         if np.sum(regionarr > mean_light + standard_var_light) > num_thresh:
#             print('this block is so bright, goodbye')
#             remove_index.append(index)
#     newlabels=[]
#     for k in range(len(labels)):
#         if k not in remove_index:
#             newlabels.append(labels[k])
#     i['labels']=newlabels
#     new_result.append(i)
# with open(result_path, 'w') as jsonf:
#     json.dump(new_result, jsonf)
#         # side1 = abs(int(x3 - x1))
#         # side2 = abs(int(y3 - y1))
#         # ratio = float(max(side1, side2) / min(side1, side2))
#         # # if (side1<50 and side2 <50) or (side1>300 or side2>300) or ratio>2.0:   # 排除过小、过大、比例失调的结果
#         # #     continue
#         # if (side1<50 and side2 <50) or (side1>300 or side2>300) or ratio>2.0:   # 排除过小、过大、比例失调的结果
#         #     continue


# # # # 针对paddle的S2ANet创建json标签
# origin_path = r'E:\datasets\832\all-r15n-832v2'
# imgpath = os.path.join(origin_path, 'img')
# maskpath = os.path.join(origin_path, 'mask')
# final_dump = {}
# final_dump["categories"] = [{"id": 1, "name": "bigship", "supercategory": "none"}]
# final_dump["images"]=[]
# final_dump["annotations"]=[]
# imglist=os.listdir(imgpath)
# masklist=os.listdir(maskpath)
# for i in imglist:
#     imgdict={}
#     name=os.path.splitext(i)[0]
#     img=Image.open(os.path.join(imgpath, i))
#     width, height=img.size
#     imgdict["file_name"]=i
#     imgdict["height"]=height
#     imgdict["id"]=name
#     imgdict["width"]=width
#     final_dump["images"].append(imgdict)
#
#
# id=0
# for i in masklist:
#     with open(os.path.join(maskpath, i), 'r') as f:
#         lines=f.readlines()
#     for line in lines:
#         id+=1
#         maskdict={}
#         name=os.path.splitext(i)[0]
#         points=line.split(' ')
#         x1, y1, x2, y2, x3, y3, x4, y4=np.float32(points[0]), np.float32(points[1]),np.float32(points[2]),np.float32(points[3]),np.float32(points[4]),np.float32(points[5]),np.float32(points[6]),np.float32(points[7])
#         cnt=np.array([[x1, y1],[x2, y2],[x3, y3],[x4, y4]])
#         rect=cv2.minAreaRect(cnt)
#         x0=int(rect[0][0])
#         y0=int(rect[0][1])
#         width=int(rect[1][0])
#         height=int(rect[1][1])
#         angle=rect[2]*math.pi/180
#         four_points=cv2.boxPoints(rect)
#         area = height*width
#
#
#         maskdict["area"]=area
#         maskdict["bbox"]=[x0, y0, width, height, angle]
#         maskdict["category_id"]=1
#         maskdict["id"]=id
#         if area==0:
#             maskdict["ignore"] = 1
#         else:
#             maskdict["ignore"]=0
#         maskdict["image_id"]=name
#         maskdict["iscrowd"]=0
#         maskdict["segmentation"]=[]
#         final_dump["annotations"].append(maskdict)
#
# with open(os.path.join(origin_path, 'train.json'), 'w') as j:
#     json.dump(final_dump, j)


# # # # # 扩大预测出的box
# jsonpath='./ship_results.json'
# n=0
# ratio=[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# newresult=[]
# with open(jsonpath, 'r') as j:
#     original=json.load(j)
# for imgresult in original:
#
#     newimgresult={}
#     name=imgresult["image_name"]
#     labels=imgresult["labels"]
#     newimgresult["image_name"]=name
#     labeltmp=[]
#     print(name)
#     for i in labels:
#         labeltmp.append(i)
#         n+=1
#         print(n)
#         conf=i["confidence"]
#         pointsi=i["points"]
#         x1,y1,x2,y2,x3,y3,x4,y4=pointsi[0][0],pointsi[0][1],pointsi[1][0],pointsi[1][1],pointsi[2][0],pointsi[2][1],pointsi[3][0],pointsi[3][1],
#         k1=(y3-y1)/(x3-x1)
#         k2=(y4-y2)/(x4-x2)
#         b1=y1-k1*x1
#         b2=y2-k2*x2
#         # 获得中心点坐标(x0,y0)
#         x0=round((b2-b1)/(k1-k2),1)
#         y0=round((k1*b2-k2*b1)/(k1-k2),1)
#         for k in ratio:
#             lineratio=math.sqrt(k)
#             labelsk={}
#             newx1=round(x0-(x0-x1)*lineratio,1)
#             newx2=round(x0-(x0-x2)*lineratio,1)
#             newx3=round(x0-(x0-x3)*lineratio,1)
#             newx4=round(x0-(x0-x4)*lineratio,1)
#             newy1=round(y0-(y0-y1)*lineratio,1)
#             newy2=round(y0-(y0-y2)*lineratio,1)
#             newy3=round(y0-(y0-y3)*lineratio,1)
#             newy4=round(y0-(y0-y4)*lineratio,1)
#             labelsk["category_id"]="bigship"
#             labelsk["points"]=[[newx1, newy1],[newx2, newy2],[newx3, newy3],[newx4, newy4]]
#             labelsk["confidence"]=conf
#             labeltmp.append(labelsk)
#     newimgresult["labels"]=labeltmp
#     newresult.append(newimgresult)
#
# with open(jsonpath, 'w') as j:
#     json.dump(newresult, j)



