from sklearn.decomposition import PCA

def pca_process(n_components):
    # n_components 选择要下降到多少维度
    X = PCA(n_components=2).fit_transform(X)
    return X