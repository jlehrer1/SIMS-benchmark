import scanpy as sc 
import matplotlib.pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import pandas as pd 
import anndata as an

def plot_allen_human():
    human = an.read_h5ad('data/benchmark/human.h5ad')
    human_labels = pd.read_csv('data/benchmark/human_labels_clean.csv')

    human = human[human_labels['cell'].values, :]
    human.obs = human.obs.reset_index() # Since we selected human_labels['cell']
    human.obs['class'] = human_labels['categorical_subclass_label']

    # Calculate PCA for UMAP
    sc.tl.pca(human, svd_solver='arpack')

    # Neighborhood embedding + umap
    sc.pp.neighbors(human, n_pcs=50)
    sc.tl.umap(human)

    # Visualization
    sc.pl.umap(human, color='class')
    
    human.write_h5ad('data/benchmark/human.h5ad')

def plot_allen_mouse():
    mouse = an.read_h5ad('data/benchmark/mouse_clipped.h5ad')
    mouse_labels = pd.read_csv('data/benchmark/mouse_labels_clean.csv')

    mouse = mouse[mouse_labels['cell'].values, :]
    mouse.obs = mouse.obs.reset_index() # Since we selected human_labels['cell']
    mouse.obs['class'] = mouse_labels['categorical_subclass_label']

    # Calculate PCA for UMAP
    sc.tl.pca(mouse, svd_solver='arpack')

    # Neighborhood embedding + umap
    sc.pp.neighbors(mouse, n_pcs=50)
    sc.tl.umap(mouse)

    # Visualization
    sc.pl.umap(mouse, color='class', save='_allen_mouse.pdf')

    # Save the new anndata object with the umap/pca coordinates
    mouse.write_h5ad('data/benchmark/mouse_clipped.h5ad')

def plot_bhaduri_human():
    bhaduri = an.read_h5ad('data/bhaduri/primary_T.h5ad')
    bhaduri_labels = pd.read_csv('data/bhaduri/primary_labels_clean.csv')

    bhaduri = bhaduri[bhaduri_labels['cell'].values, :]
    bhaduri.obs = bhaduri.obs.reset_index() # Since we selected human_labels['cell']
    bhaduri.obs['class'] = bhaduri_labels['categorical_Subtype']

    # Calculate PCA for UMAP
    sc.tl.pca(bhaduri, svd_solver='arpack')

    # Neighborhood embedding + umap
    sc.pp.neighbors(bhaduri, n_pcs=50)
    sc.tl.umap(bhaduri)

    # Visualization
    sc.pl.umap(bhaduri, color='class', save='_bhaduri_cortical.pdf')

if __name__ == "__main__":
    plot_allen_human()
    plot_allen_mouse()
    plot_bhaduri_human()