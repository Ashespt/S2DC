import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE,SpectralEmbedding,Isomap,LocallyLinearEmbedding
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
import umap
tsne = TSNE(n_components=3, random_state=5,perplexity=100)
pca = PCA(n_components=5)
isomap = Isomap(n_components=3, n_neighbors=10)
llm = LocallyLinearEmbedding(n_components=3)
umap = umap.UMAP()
tsne3d = TSNE(n_components=3, random_state=50, perplexity=30)
pca3d = PCA(n_components=3)
isomap3d = Isomap(n_components=3, n_neighbors=10)
SE = SpectralEmbedding(n_components=2,n_neighbors=10)

def pcatsne2d_visual(features,labels,save_path,method='pca',wolabel=1):
    features=features[labels!=wolabel]
    labels=labels[labels!=wolabel]
    # labels[labels==2] = 1
    if method=='pca':
        data_2d = pca.fit_transform(features)
    elif method == 'tsne':
        data_2d = tsne.fit_transform(features)
    elif method == 'isomap':
        data_2d = isomap.fit_transform(features)
    elif method == 'llm':
        data_2d = llm.fit_transform(features)
    elif method == 'umap':
        data_2d = umap.fit_transform(features)
    colors = ['orange', 'darkturquoise','cornflowerblue']
    label_names = ['NC','CP','NCP']
    total_labels = np.array([0,1,2])
    for i in total_labels[total_labels!=wolabel]:
        color = colors[i]
        subset = data_2d[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    plt.legend()
    # plt.title('t-SNE visualization of validation dataset')
    # plt.xlabel('t-SNE feature 1')
    # plt.ylabel('t-SNE feature 2')
    plt.savefig(save_path)
    plt.close()

def pcatsne3d_visual(features,labels,save_path,method='pca',wolabel=1):
    features=features[labels!=wolabel]
    labels=labels[labels!=wolabel]
    if method == 'pca':
        data_3d = pca3d.fit_transform(features)
    elif method == 'tsne':
        data_3d = tsne3d.fit_transform(features)
    elif method == 'isomap':
        data_3d = isomap3d.fit_transform(features)
    colors = ['r', 'b','g']
    label_names = ['NC','CP','NCP']
    total_labels = np.array([0,1,2])

    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection='3d')  # 设置为3d图
    
    colors = ['r', 'b','g']
    label_names = ['NC','CP','NCP']
    total_labels = np.array([0,1,2])
    for i in total_labels[total_labels!=wolabel]:
        color = colors[i]
        subset = data_3d[labels == i]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=color)
    plt.savefig(save_path)
    plt.close()

def SE_visual(features,labels,save_path,wolabel=0):
    features=features[labels!=wolabel]
    labels=labels[labels!=wolabel]
    data_2d = SE.fit_transform(features)
    colors = ['r', 'b','g']
    label_names = ['NC','CP','NCP']
    total_labels = np.array([0,1,2])
    for i in total_labels[total_labels!=wolabel]:
        color = colors[i]
        subset = data_2d[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    plt.legend()
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter)
    plt.title('SE visualization of validation dataset')
    plt.xlabel('SE feature 1')
    plt.ylabel('SE feature 2')
    plt.savefig(save_path)
    plt.close()
