#把自己的图片转为VOC的组织方式，方便计算
import os
import shutil
import glob
from xml.dom.minidom import Document
import random
import scipy.ndimage
import re

#txt转xml
def write_info_to_Xml(img_name,img_size,spts,file_name,datebase_name="ICDAR"):
    #创建dom文档
    doc = Document()
    #创建根节点
    annotation = doc.createElement("annotation")
    #插入根节点dom树
    doc.appendChild(annotation)
    #folder
    folder = doc.createElement("folder")
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)
    annotation.appendChild(folder)
    #filename
    filename = doc.createElement("filename")
    filename_txt = doc.createTextNode(img_name)#这里是xx.jpg
    filename.appendChild(filename_txt)
    annotation.appendChild(filename)
    #<source>
    source = doc.createElement("source")
    #database
    database = doc.createElement("database")
    database_txt = doc.createTextNode(datebase_name)
    database.appendChild(database_txt)
    source.appendChild(database)
    #annotation
    annotation_source = doc.createElement("annotation")
    annotation_txt = doc.createTextNode("PASCAL VOC2007")
    annotation_source.appendChild(annotation_txt)
    source.appendChild(annotation_source)
    #image
    image = doc.createElement("image")
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    source.appendChild(image)
    #flickrid
    flickrid = doc.createElement("flickrid")
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)
    source.appendChild(flickrid)
    annotation.appendChild(source)
    #<source>结束
    #<owner>
    owner = doc.createElement("owner")
    #flickrid
    flickrid = doc.createElement("flickrid")
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)
    owner.appendChild(flickrid)
    #name
    name = doc.createElement("name")
    name_txt = doc.createTextNode("owner")
    name.appendChild(name_txt)
    owner.appendChild(name)
    annotation.appendChild(owner)
    #owner结束
    #size
    size = doc.createElement("size")
    # flickrid
    width = doc.createElement("width")
    width_txt = doc.createTextNode(str(img_size[0]))
    width.appendChild(width_txt)
    size.appendChild(width)
    hight = doc.createElement("height")
    hight_txt = doc.createTextNode(str(img_size[1]))
    hight.appendChild(hight_txt)
    size.appendChild(hight)
    deep = doc.createElement("depth")
    deep_txt = doc.createTextNode("3")
    deep.appendChild(deep_txt)
    size.appendChild(deep)

    annotation.appendChild(size)
    #size结束
    #<segmented>0</segmented>
    segmented = doc.createElement("segmented")
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)
    annotation.appendChild(segmented)
    #object
    for spt in spts:
        object = doc.createElement("object")
        #name
        objName = doc.createElement("name")
        objName_txt = doc.createTextNode("words")
        objName.appendChild(objName_txt)
        object.appendChild(objName)
        #pose
        pose = doc.createElement("pose")
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)
        object.appendChild(pose)
        #truncated
        truncated = doc.createElement("truncated")
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)
        object.appendChild(truncated)
        #difficult
        difficult = doc.createElement("difficult")
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        object.appendChild(difficult)
        #bndbox
        bndbox = doc.createElement("bndbox")
        # xmin
        xmin = doc.createElement("xmin")
        xmin_txt = doc.createTextNode(spt[0])
        xmin.appendChild(xmin_txt)
        bndbox.appendChild(xmin)
        # xmin
        ymin = doc.createElement("ymin")
        ymin_txt = doc.createTextNode(spt[1])
        ymin.appendChild(ymin_txt)
        bndbox.appendChild(ymin)
        # xmax
        xmax = doc.createElement("xmax")
        xmax_txt = doc.createTextNode(spt[2])
        xmax.appendChild(xmax_txt)
        bndbox.appendChild(xmax)
        # ymax
        ymax = doc.createElement("ymax")
        ymax_txt = doc.createTextNode(spt[3])
        ymax.appendChild(ymax_txt)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)
    with open(file_name, 'wb+') as f:
        f.write(doc.toprettyxml(indent='\t',encoding='utf-8'))
    return

#创建txt
def crate_txt(count,fileDir,test_num):
    testFile = open(os.path.join(fileDir, "ssd_test.txt"),'w')
    trainFile = open(os.path.join(fileDir, "train.txt"),'w')
    voc_list = range(count)
    voc_testID = sorted(random.sample(voc_list,test_num))
    all_file = []
    for i in voc_list:
        all_file.append(i)
    for i in voc_testID:
        all_file.remove(i)
        testFile.write(str("%06d"%i)+'\n')
    for i in all_file:
        trainFile.write(str("%06d"%i)+'\n')
    testFile.close()
    trainFile.close()
    return


def get_img_size(img_dir):
    #img = Image.open(img_dir,mode='r')
    #img_size = img.size
    img_size = scipy.ndimage.imread(img_dir).shape
    return img_size

def save_img(img_get_dir,img_put_dir):
    shutil.copyfile(img_get_dir, img_put_dir)
    pass

def change_data(image_dir,gt_text_dir,imgs_save_dir,save_text_dir,count = 0):
    imgDirs = []
    imgLists = glob.glob(image_dir)#返回一个list，里面都是文件名

    for item in imgLists:
        imgDirs.append(item)
    for img_dir in imgDirs:
        #img = Image.open(img_dir)
        img_size = get_img_size(img_dir)

        img_basename = os.path.basename(img_dir)#就是文件名xx.jpg
        (img_name, prefix) = os.path.splitext(img_basename)#img_name就是xx，prefix 是 .jpg
        save_img_name = str('%06d'%count)+prefix
        # open the ground truth text file
        img_gt_text_name = "gt_" + img_name + ".txt"
        save_xml_name = str('%06d.xml'%count)
        print(img_gt_text_name)

        bf = open(os.path.join(gt_text_dir, img_gt_text_name)).read().splitlines()
        spts = []
        for idx in bf:
            rect = []
            spt = re.split(' |, ', idx)
            print(spt)
            rect.append(spt[0])
            rect.append(spt[1])
            rect.append(spt[2])
            rect.append(spt[3])
            spts.append(rect)
        write_info_to_Xml(save_img_name, img_size, spts, os.path.join(save_text_dir, save_xml_name))

        save_img(img_dir,os.path.join(imgs_save_dir, save_img_name))

        count = count +1
    return count

# ground truth directory
gt_text_dir1 = "ssd_data/Challenge2_Training_Task1_GT"
save_text_dir1 = "ssd_data/VOC2007/Annotations"
# original images directory
image_dir1 = "ssd_data/Challenge2_Training_Task12_Images/*.jpg"
imgs_save_dir1 = "ssd_data/VOC2007/JPEGImages"
count = change_data(image_dir1,gt_text_dir1,imgs_save_dir1,save_text_dir1)
# ground truth directory
gt_text_dir2 = "ssd_data/Challenge2_Test_Task1_GT"
# original images directory
image_dir2 = "ssd_data/Challenge2_Test_Task12_Images/*.jpg"
count = change_data(image_dir2,gt_text_dir2,imgs_save_dir1,save_text_dir1,count)
txt_file_dir = "ssd_data/VOC2007/ImageSets/Main"
crate_txt(count,txt_file_dir,233)