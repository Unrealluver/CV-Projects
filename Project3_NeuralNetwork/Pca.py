from sklearn.decomposition import PCA

def pca_process(X, n_components):
    # n_components 选择要下降到多少维度
    X = PCA(n_components=n_components).fit_transform(X)
    return X