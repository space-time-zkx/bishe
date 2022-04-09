import os
import cv2
with open("/data/zhaokexin/vision/3d_object/patchnet-master/2d_results/"+"000009.txt","r")as f:
    img = "/data/zhaokexin/dataset/KITTI_3dobject/object/testing/image_2/000009.png"
    img = cv2.imread(img)
    datas = f.readlines()
    for data in datas:
        if data.split(" ")[0]=="Car":
            cv2.rectangle(img,(int(data.split(" ")[4]),int(data.split(" ")[5])),(int(data.split(" ")[6]),int(data.split(" ")[7])),color=(255,0,0))
            cv2.putText(img,"Car",(int(data.split(" ")[4]), int(data.split(" ")[5])),2,
                    cv2.FONT_HERSHEY_SIMPLEX,(255,0,0),2)
            print("car")
        elif data.split(" ")[0]=="Pedestrian":
            cv2.rectangle(img, (int(data.split(" ")[4]), int(data.split(" ")[5])),
                          (int(data.split(" ")[6]), int(data.split(" ")[7])), color=(255,255,0))
            cv2.putText(img,"Pede",(int(data.split(" ")[4]), int(data.split(" ")[5])),2,
                           cv2.FONT_HERSHEY_SIMPLEX ,(255,255,0),2)
            print("ped")
        else:
            cv2.rectangle(img, (int(data.split(" ")[4]), int(data.split(" ")[5])),
                          (int(data.split(" ")[6]), int(data.split(" ")[7])), color=(0,255,0))
            cv2.putText(img,"Cyc",(int(data.split(" ")[4]), int(data.split(" ")[5])),2,
                           cv2.FONT_HERSHEY_SIMPLEX,(0,255,0),2)
            print("cyc")
    cv2.imwrite("2d.png",img)