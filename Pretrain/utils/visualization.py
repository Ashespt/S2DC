import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import umap
umap = umap.UMAP()
tsne = TSNE(n_components=2,perplexity=24)#, random_state=42)
pca = PCA(n_components=2)
pca_1d = PCA(n_components=1)
def visual_2d(method,features,labels,save_path):
    data_2d = tsne.fit_transform(features)
    silhouette_avg = silhouette_score(features, labels)
    print("Silhouette Score:", silhouette_avg)

    if method=='pca':
        data_2d = pca.fit_transform(features)
    elif method == 'tsne':
        data_2d = tsne.fit_transform(features)
    elif method == 'umap':
        data_2d = umap.fit_transform(features)
    # data_2d[:,1] =data_2d[:,1]/(data_2d[:,1].max()-data_2d[:,1].min())
    colors = ['xkcd:orange','xkcd:sky blue','limegreen','teal','plum','saddlebrown','b', 'g','r','c','m','y','k','w']
    label_names = ['Spl','RKid','LKid','Gall','Eso','Liv','Sto','Aor','IVC','Veins','Pan','RAG','LAG']
    for i, color in enumerate(colors[:labels.max()]):
        subset = data_2d[labels == i+1]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter)
    plt.title(f'method')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig(save_path)
    plt.close()

def tsne_visual(features,labels,save_path):
    data_2d = tsne.fit_transform(features)
    colors = ['xkcd:orange','xkcd:sky blue','limegreen','tab:brown','plum','saddlebrown','b', 'g','r','c','m','y','k','w']
    label_names = ['Spl','RKid','LKid','Gall','Eso','Liv','Sto','Aor','IVC','Veins','Pan','RAG','LAG']
    for i, color in enumerate(colors[:labels.max()]):
        subset = data_2d[labels == i+1]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter)
    plt.title('t-SNE')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig(save_path)
    plt.close()

def pca_visual(features,labels,save_path):
    data_2d = pca.fit_transform(features)
    colors = ['xkcd:orange','xkcd:sky blue','limegreen','tab:brown','plum','saddlebrown','b', 'g','r','c','m','y','k','w']
    label_names = ['Spl','RKid','LKid','Gall','Eso','Liv','Sto','Aor','IVC','Veins','Pan','RAG','LAG']
    for i, color in enumerate(colors[:labels.max()]):
        subset = data_2d[labels == i+1]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.title('PCA visualization of validation dataset')
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    plt.savefig(save_path)
    plt.close()

import matplotlib as mpl
from matplotlib import cm
import nibabel as nib
cmap_name = 'cividis'
# cmap_name = 'Pastel1'
def pca_1d_visual(features,save_path,dim=192):
    features = features.reshape(-1,dim)
    data_1d = pca_1d.fit_transform(features)
    t = features.shape[0]
    t = int(np.ceil(math.pow(t/16, 1/3)))
    data_1d = data_1d.reshape(t*4,t*4,-1)
    d = data_1d.shape[-1]
    
    new_image = nib.Nifti1Image(data_1d, np.eye(4)) 
    new_image.set_data_dtype(np.float32) 
    
    # nib.save(new_image, save_path)  
    #slice
    data = data_1d[:,:,int(d//2)]
    plt.imshow(data, cmap=cmap_name)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize()
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = scalarMap.to_rgba(data)

    image = Image.fromarray((colors[:, :, :3]*256).astype(np.uint8))
    image.save(save_path.replace('.nii.gz','.png'))
    plt.close()