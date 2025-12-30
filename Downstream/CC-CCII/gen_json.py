import json
import pandas as pd
import pdb
import os
from collections import OrderedDict
import numpy as np
json_dict = OrderedDict()
json_dict['training'] = []
json_dict['validation'] = []
json_dict['name'] = "CC-CCII"
json_dict['tensorImageSize'] = "3D"
json_dict['release'] = "0.0"
json_dict['training'] = []
json_dict['validation'] = []
json_dict['modality'] = {
        "0": "CT"
    }
# all_paths = ['/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/npy_new/p1036-s2607.npy',
#             '']
a = pd.read_csv('csv/CC_CCII_fold0_train.csv')
df = a[a['target']==2]
patients = np.array(df['patient_id'])
scans = np.array(df['scan_id'])
targets = np.array(df['target'])
data_dir = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/ct-opensource/CC-CCII/npy_new/'
with open('visual_NCP.json','w') as f:
    for index in range(len(targets)):
        path = os.path.join(
                data_dir,
                'p'+str(patients[index])+'-s'+str(scans[index])+'.npy'
                )
        json_dict['training'].append({'image':path})
    json.dump(json_dict,f)