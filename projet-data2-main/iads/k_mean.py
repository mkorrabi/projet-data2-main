# Import de packages externes
import numpy as np
import pandas as pd
import numpy as np
import copy


def centroide(data):
    return data.mean(axis=0)





def dist_vect(v1, v2):
    return clust.dist_euclidienne(v1, v2)



def inertie_cluster(Ens):
    centroide = clust.centroide(Ens)
    Ens = np.asarray(Ens)
    inertie = 0
    for i in range(len(Ens)):
        inertie += dist_vect(Ens[i], centroide)**2

    return inertie
    
############# A COMPLETER 


import random 
def init_kmeans(K,Ens):
    Ens = np.asarray(Ens)
    return Ens[np.random.choice(Ens.shape[0], K, replace=False), :]
    
############# A COMPLETER 


def plus_proche(Exe,Centres):
    mini=99999999999
    index=-1
    for c in range(len(Centres)) :
        if  (dist_vect(Exe, Centres[c]) ) < mini :
            mini = dist_vect(Exe, Centres[c])
            index = c
    return index
# ############# A COMPLETER 
# def plus_proche(Exe,Centres):
#     mini=999999999999
#     indice=0
#     for i in range(len(Centres)):
#         if dist_vect(Exe,Centres[i])<mini:
#             mini=dist_vect(Exe,Centres[i])
#             indice=i
#     return i
            
        
############# A COMPLETER 

def affecte_cluster(Base,Centres):
    U = {i: [] for i in range(len(Centres))}
    Base = np.asarray(Base)
    for i in range(len(Base)):
        U[plus_proche(Base[i], Centres)].append(i)
    return U
    
############# A COMPLETER 

def nouveaux_centroides(Base,U):
    t = []
    Base = np.asarray(Base)
    for i in U.keys():
        t.append(clust.centroide(Base[U[i]]))
    return np.asarray(t)

############# A COMPLETER 


def inertie_globale(Base, U):
    inertie = 0
    Base = np.asarray(Base)
    for i in U.keys():
        inertie += inertie_cluster(Base[U[i]])
    return inertie
############# A COMPLETER 
def kmoyennes(K, Base, epsilon, iter_max):
    Centroides = init_kmeans(K, Base)
    # Affectation des exemples à chaque cluster
    U = affecte_cluster(Base, Centroides)
    # Calcul de l'inertie globale
    inertie = inertie_globale(Base, U)
    print(inertie)
    for i in range(iter_max):
        Nouveaux_Centroides = nouveaux_centroides(Base, U)
        Centroides = Nouveaux_Centroides
        Nouveaux_U = affecte_cluster(Base, Centroides)
        Nouvelle_Inertie = inertie_globale(Base, Nouveaux_U)
        print("iteration",i+1," Inertie",Nouvelle_Inertie, "Différence", f'{abs(Nouvelle_Inertie-inertie):1.4f}')
        U = Nouveaux_U
        if abs(Nouvelle_Inertie - inertie) < epsilon:
            break
        inertie = Nouvelle_Inertie
    return Centroides, U
############# A COMPLETER 
