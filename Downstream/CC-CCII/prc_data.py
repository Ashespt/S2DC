import cv2
import os
import numpy as np
from tqdm import tqdm
path1 = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/NCP/NCP'
path2 = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/CP/CP'
path3 = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/NC/Normal'
save_path = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/npy_new/'

def worker(path):
# for path in [path1,path2,path3]:
    # for p in tqdm(os.listdir(path)):
    # for s in os.listdir(os.path.join(path,p)):
    for p in tqdm(path):
        for s in os.listdir(p):
            try:
                images = os.listdir(os.path.join(p,s))
                images.sort(key=lambda x:int(x.split('.')[0]))
                
                images_list = []
                for image in images:
                    image = cv2.imread(os.path.join(p,s,image))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image,[512,512])
                    images_list.append(image)
                image = np.transpose(np.stack(images_list),(1,2,0))
                p_n = p.split('/')[-1]
                save_path_image = save_path+'p'+p_n+'-s'+s
                np.save(save_path_image,image)
            except:
                print(os.path.join(p,s))
                with open('failed_image.txt','a') as f:
                    f.write(os.path.join(p,s)+'\n')
paths = [path1,path2,path3]
path_list = []
for path in paths:
    for p in os.listdir(path):
        path_list.append(os.path.join(path,p))
import multiprocessing
import pdb;pdb.set_trace()
import math
jobs = []
num_worker = 10
space = math.ceil(len(path_list)/num_worker)
# worker(path_list)
for i in range(num_worker):
    p = multiprocessing.Process(target=worker, args=(path_list[i*space:(i+1)*space],))
    jobs.append(p)
    p.start()
for job in jobs:
    job.join()

# with open('failed_image.txt') as f:
#     for path in f.readlines():
#         try:
#             path =path.split('\n')[0]
#     # path = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/NC/Normal/1748/1066'
#             images = os.listdir(path)
#             images_list = []
#             for image in images:
#                 images_list.append(cv2.resize(cv2.imread(os.path.join(path,image))[...,0],[256,256]))
#             image = np.stack(images_list)
#             p,s=path.split('/')[-2:]
            
#             save_path = f'/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/npy/p{p}-s{s}.npy'
#             np.save(save_path,image)
#         except:
#             print(path)

