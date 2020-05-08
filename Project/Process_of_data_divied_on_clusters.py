# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:16:56 2020

@author: adely
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import seaborn as sns
import scipy
from math import cos
from matplotlib.pyplot import scatter



data_0 = pd.read_csv('clusterized data\class_0_by_difficulty.csv')
data_1 = pd.read_csv('clusterized data\class_1_by_difficulty.csv')
data_2 = pd.read_csv('clusterized data\class_2_by_difficulty.csv')
data_3 = pd.read_csv('clusterized data\class_3_by_difficulty.csv')
data_4 = pd.read_csv('clusterized data\class_4_by_difficulty.csv')
data_5 = pd.read_csv('clusterized data\class_5_by_difficulty.csv')
data_6 = pd.read_csv('clusterized data\class_6_by_difficulty.csv')
data_7 = pd.read_csv('clusterized data\class_7_by_difficulty.csv')
data_8 = pd.read_csv('clusterized data\class_8_by_difficulty.csv')
col_0 = data_0[['88.0.1', '68.55']]
col_1 = data_1[['74.0.1', '50.27']]
col_2 = data_2[['197.0.1', '81.45']]
col_3 = data_3[['86.0.1', '71.05']]
col_4 = data_4[['37.0.1', '36.15']]
col_5 = data_5[['76.0.1', '55.17']]
col_6 = data_6[['41.0.1', '39.65']]
col_7 = data_7[['119.0.1', '79.4']]
col_8 = data_8[['43.0.1', '40.48']]
ax0=col_0.plot.scatter(x = '68.55', y = '88.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 1')
ax0.set_ylabel("Bestfit in the evoluation with state parameters")
ax0.set_xlabel("Average fitness in the evoluation with state parameters")

ax1=col_1.plot.scatter(x = '50.27', y = '74.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 2')
ax1.set_ylabel("Bestfit in the evoluation with state parameters")
ax1.set_xlabel("Average fitness in the evoluation with state parameters")

ax2=col_2.plot.scatter(x = '81.45', y = '197.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 3')
ax2.set_ylabel("Bestfit in the evoluation with state parameters")
ax2.set_xlabel("Average fitness in the evoluation with state parameters")

ax3=col_3.plot.scatter(x = '71.05', y = '86.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 4')
ax3.set_ylabel("Bestfit in the evoluation with state parameters")
ax3.set_xlabel("Average fitness in the evoluation with state parameters")

ax4=col_4.plot.scatter(x = '36.15', y = '37.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 5')
ax4.set_ylabel("Bestfit in the evoluation with state parameters")
ax4.set_xlabel("Average fitness in the evoluation with state parameters")

ax5=col_5.plot.scatter(x = '55.17', y = '76.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 6')
ax5.set_ylabel("Bestfit in the evoluation with state parameters")
ax5.set_xlabel("Average fitness in the evoluation with state parameters")

ax6=col_6.plot.scatter(x = '39.65', y = '41.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 7')
ax6.set_ylabel("Bestfit in the evoluation with state parameters")
ax6.set_xlabel("Average fitness in the evoluation with state parameters")

ax7=col_7.plot.scatter(x = '79.4', y = '119.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 8')
ax7.set_ylabel("Bestfit in the evoluation with state parameters")
ax7.set_xlabel("Average fitness in the evoluation with state parameters")

ax8=col_8.plot.scatter(x = '40.48', y = '43.0.1', title = 'The graph of dependence bestfit over average fitness in the cluster 9')
ax8.set_ylabel("Bestfit in the evoluation with state parameters")
ax8.set_xlabel("Average fitness in the evoluation with state parameters")




















data = pd.read_csv('environment_results_data.csv')
#data = data.sort_values(by=['Iteration'])
# =============================================================================
# data_state_1 = data[data['State 1']==-1.944]
# data_state_1.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_1 = data[data['State 1']==-0.972]
# data_state_1.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_1 = data[data['State 1']==0]
# data_state_1.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_1 = data[data['State 1']==0.972]
# data_state_1.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_1 = data[data['State 1']==1.944]
# data_state_1.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# data_state_2 = data[data['State 2']==-1.215]
# data_state_2.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_2 = data[data['State 2']==-0.6075]
# data_state_2.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_2 = data[data['State 2']==0]
# data_state_2.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_2 = data[data['State 2']==0.6075]
# data_state_2.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_2 = data[data['State 2']==1.215]
# data_state_2.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# data_state_3 = data[data['State 3']==-0.10472]
# data_state_3.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_3 = data[data['State 3']==-0.05236]
# data_state_3.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_3 = data[data['State 3']==0]
# data_state_3.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_3 = data[data['State 3']==0.05236]
# data_state_3.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_3 = data[data['State 3']==0.10472]
# data_state_3.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# data_state_4 = data[data['State 4']== -0.135088]
# data_state_4.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_4 = data[data['State 4']== -0.067544]
# data_state_4.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_4 = data[data['State 4']== 0]
# data_state_4.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_4 = data[data['State 4']== 0.067544]
# data_state_4.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_4 = data[data['State 4']== 0.135088]
# data_state_4.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# data_state_5 = data[data['State 5']== -0.10472]
# data_state_5.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_5 = data[data['State 5']== -0.05236]
# data_state_5.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_5 = data[data['State 5']== 0]
# data_state_5.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_5 = data[data['State 5']== 0.05236]
# data_state_5.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_5 = data[data['State 5']== 0.10472]
# data_state_5.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# data_state_6 = data[data['State 6']== -0.135088]
# data_state_6.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_6 = data[data['State 6']== -0.067544]
# data_state_6.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_6 = data[data['State 6']== 0]
# data_state_6.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_6 = data[data['State 6']== 0.067544]
# data_state_6.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# data_state_6 = data[data['State 6']== -0.135088]
# data_state_6.plot(x = 'avg', y = 'bestgfit', kind = 'scatter')
# =============================================================================

# =============================================================================
# def calculate_energy_init_state(state):
# 	GRAVITY = 9.8
# 	MASSCART = 1.0	
# 	M_1 = 0.1
# 	M_2 = 0.05;
# 	L_1 = 0.5
# 	L_2 = 0.25;
# 	I_1=1/3*M_1*L_1**2
# 	I_2=1/3*M_2*L_2**2
# 	x=state[0]
# 	dx=state[1]
# 	theta_1=state[2]
# 	dtheta_1=state[3]
# 	theta_2=state[4]
# 	dtheta_2=state[5]
# 	
# 	K_cart=MASSCART*dx**2/2
# 	P_cart=0
# 	
# 	K_pole_1=I_1*dtheta_1**2/2
# 	P_pole_1=M_1*GRAVITY*cos(theta_1)*L_1/2
# 
# 	K_pole_2=I_2*dtheta_2**2/2
# 	P_pole_2=M_2*GRAVITY*cos(theta_2)*L_2/2
# 	
# 	return K_cart+P_cart+K_pole_1+P_pole_1+K_pole_2+P_pole_1
# 
# 
# 
# potential_energy = []
# state = data[['State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6']]
# state = np.asarray(state)
# 
# for i in range(state.shape[0]):
#     potential_energy.append(calculate_energy_init_state(state[i,:]))
#     
# bestfit = data[['bestgfit']]
# bestfit = np.asarray(bestfit)
# 
# scatter(potential_energy, bestfit)
# =============================================================================


#data_cluster = data[['State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6','bestgfit']]

#no dependencies were found
# =============================================================================
# h = sns.clustermap(data_cluster, metric = 'correlation', standard_scale=1)
# den = scipy.cluster.hierarchy.dendrogram(h.dendrogram_col.linkage,
#                                          labels = data_cluster.index,
#                                          color_threshold=0.10) 
# 
# from collections import defaultdict
# 
# def get_cluster_classes(den, label='ivl'):
#     cluster_idxs = defaultdict(list)
#     for c, pi in zip(den['color_list'], den['icoord']):
#         for leg in pi[1:3]:
#             i = (leg - 5.0) / 10.0
#             if abs(i - int(i)) < 1e-5:
#                 cluster_idxs[c].append(int(i))
# 
#     cluster_classes = {}
#     for c, l in cluster_idxs.items():
#         i_l = [den[label][i] for i in l]
#         cluster_classes[c] = i_l
# 
#     return cluster_classes
# 
# clusters = get_cluster_classes(den)
# 
# cluster = []
# for i in data_cluster.index:
#     included=False
#     for j in clusters.keys():
#         if i in clusters[j]:
#             cluster.append(j)
#             included=True
#     if not included:
#         cluster.append(None)
#         
# print(cluster)
# 
# print(h.dendrogram_row.linkage, h.dendrogram_row.linkage)
# 
# dgram = h.dendrogram_row.dendrogram
# 
# 
# 
# D = np.array(dgram['dcoord'])
# I = np.array(dgram['icoord'])
# 
# yy = D[-1] 
# lenL = yy[1]-yy[0]
# lenR = yy[2]-yy[3]
# =============================================================================


