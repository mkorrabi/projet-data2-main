# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import pandas as pd
import numpy as np
import copy

# ---------------------------
# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        prediction = []
        for i in range(len(desc_set)):
            prediction.append(self.predict(desc_set[i]))
        valide = 0
        for i in range(len(prediction)):
            if(prediction[i] == label_set[i]):
                valide += 1
        return valide / len(prediction)
        
# ---------------------------
# ------------------------ A COMPLETER :
import math

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        self.eucl_dist_tab = np.zeros((self.desc_set.shape[0]))
        for i in range(len(self.desc_set)):
            eucl_dist_sum = 0
            for j in range(len(self.desc_set[0])):
                eucl_dist_sum += math.pow(x[j]-self.desc_set[i][j],2)
            self.eucl_dist_tab[i] = math.sqrt(eucl_dist_sum)
        self.sorted_arg_eucl_dist = np.argsort(self.eucl_dist_tab)
        p = 0
        for i in range(self.k):
            if(self.label_set[self.sorted_arg_eucl_dist[i]] == 1):
                p += 1
        p = p / self.k
        
        return 2*(p-0.5)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score = self.score(x)
        # Ici on simplifie en disant que si le score est nul on est dans la classe 1
        if(score > 0):
            res = 1
        else:
            res = -1
        return res

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if(init == 0):
            self.w = np.zeros(input_dimension)
        elif(init == 1):
            self.w = []
            for i in range(input_dimension):
                self.w.append((np.random.uniform()*2 - 1)*0.001)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        max_index = len(desc_set)
        index_tab = []
        for i in range(max_index):
            index_tab.append(i)
        np.random.shuffle(index_tab)
        for index in index_tab:
            if(not np.sign(np.dot(desc_set[index,:],self.w)) == label_set[index]):
                self.w = self.w + self.learning_rate*desc_set[index,:]*label_set[index]
         
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        convergence_tab = []
        for i in range(niter_max):
                                    
            self.w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            temp_vect = np.absolute(self.w_before - self.w)
            convergence = 0
            for elt in temp_vect:
                convergence += elt*elt
            convergence = np.sqrt(convergence)
            convergence_tab.append(convergence)
            if(convergence < seuil):
                return convergence_tab
        
        return convergence_tab
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        score = np.dot(self.w,x)
        return score
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if(score > 0):
            res = 1
        else:
            res = -1
        return res

    
# ---------------------------
# ------------------------ A COMPLETER :
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if(init == 0):
            self.w = np.zeros(input_dimension)
        elif(init == 1):
            self.w = []
            for i in range(input_dimension):
                self.w.append((np.random.uniform()*2 - 1)*0.001)
        self.noyau = noyau
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        desc_set_ker = self.noyau.transform(desc_set)
        label_set_ker = self.noyau.transform(label_set)
        max_index = len(desc_set_ker)
        index_tab = []
        for i in range(max_index):
            index_tab.append(i)
        np.random.shuffle(index_tab)
        for index in index_tab:
            if(not np.sign(np.dot(desc_set_ker[index,:],self.w)) == label_set_ker[index]):
                self.w = self.w + self.learning_rate*desc_set_ker[index,:]*label_set_ker[index]
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        convergence_tab = []
        for i in range(niter_max):           
            self.w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            temp_vect = np.absolute(self.w_before - self.w)
            convergence = 0
            for elt in temp_vect:
                convergence += elt*elt
            convergence = np.sqrt(convergence)
            convergence_tab.append(convergence)
            if(convergence < seuil):
                return convergence_tab
        
        return convergence_tab   
    
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        score = np.dot(self.w,x)
        return score
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        score = self.score(x)
        if(score > 0):
            res = 1
        else:
            res = -1
        return res


class ClassifierPerceptronBiais(Classifier):
    def __init__(self, dimo, epsi, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.dim = dimo
        self.eps = epsi
        
        if(init == 0):
            self.w = np.zeros(self.dim)
        elif(init == 1):
            self.w = []
            for i in range(self.dim):
                self.w.append((np.random.uniform()*2 - 1)*0.001)
        self.allw=[]
        self.allw.append(self.w.copy())
        self.c=0
        self.tab_c=[]
        
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        max_index = len(desc_set)
        index_tab = []
        for i in range(max_index):
            index_tab.append(i)
        np.random.shuffle(index_tab)
        for index in index_tab:
            if(np.dot(desc_set[index,:],self.w)*label_set[index]) <1:
                self.w = self.w + self.eps*(label_set[index]-(self.score(desc_set[index,:])))*desc_set[index,:]
                self.allw.append(self.w.copy())
                if (np.dot(desc_set[index,:],self.w)*label_set[index])<=0:
                    self.c=self.c+(1-0)
                else:
                    self.c=self.c+(1-(np.dot(desc_set[index,:],self.w)*label_set[index]))
                self.tab_c.append(self.c)
        for i in range(0,len(self.tab_c)):
            self.tab_c[i]=self.c-self.tab_c[i]
            
    def get_allw(self):
        return self.allw
    def get_c(self):
        return self.c
    def get_tab_c(self):
        return self.tab_c
    
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        convergence_tab = []
        for i in range(niter_max):
                                    
            self.w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            temp_vect = np.absolute(self.w_before - self.w)
            convergence = 0
            for elt in temp_vect:
                convergence += elt*elt
            convergence = np.sqrt(convergence)
            convergence_tab.append(convergence)
            if(convergence < seuil):
                return convergence_tab
        
        return convergence_tab
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        score = np.dot(self.w,x)
        return score
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if(score > 0):
            res = 1
        else:
            res = -1
        return res
    
    
class Perceptron_MC(Classifier):
    
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if(init == 0):
            self.w = np.zeros(input_dimension)
        elif(init == 1):
            self.w = []
            for i in range(input_dimension):
                self.w.append((np.random.uniform()*2 - 1)*0.001)
        
    def train_step(self, desc_set, label_set,classe):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        """//////
        """
        label_set_2=copy.deepcopy(label_set)
        L=[]
        for i in range(0,len(label_set)):
            if label_set[i]!=classe :
                label_set_2[i]=-1
        
        """////
        """
        max_index = len(desc_set)
        index_tab = []
        for i in range(max_index):
            index_tab.append(i)
        np.random.shuffle(index_tab)
        for index in index_tab:
            if(not np.sign(np.dot(desc_set[index,:],self.w)) == label_set_2[index]):
                self.w = self.w + self.learning_rate*desc_set[index,:]*label_set_2[index]
         
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        convergence_tab = []
        L=[]
        for i in label_set:
            if i not in L:
                L.append(i)
#         print(L)



        for c in  L:  
            for i in range(niter_max):

                self.w_before = self.w.copy()
                self.train_step(desc_set,label_set,c)
                temp_vect = np.absolute(self.w_before - self.w)
                convergence = 0
                for elt in temp_vect:
                    convergence += elt*elt
                convergence = np.sqrt(convergence)
                convergence_tab.append(convergence)
                if(convergence < seuil):
                    return convergence_tab

#             return convergence_tab
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        score = np.dot(self.w,x)
        return score
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if(score > 0):
            res = 1
        elif(score < 0):
            res = -1
        return res
    
    
class ClassifierMultiOAA(Classifier):
    def __init__(self,classif_binaire):
        self.classif_binaire = classif_binaire
        self.classif_list = []
         
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        self.labels = list(set(label_set))
        for i in range(len(self.labels)):
            self.classif_list.append(copy.deepcopy(self.classif_binaire))
        for i in range(len(self.classif_list)):
            curr_label = self.labels[i]
            curr_desc_set = copy.deepcopy(desc_set)
            curr_label_set = list(1 if elt == curr_label else -1 for elt in label_set)
            self.classif_list[i].train(curr_desc_set,curr_label_set)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        score = [self.classif_list[i].score(x) for i in range(len(self.classif_list))]
        return score
    
    def predict(self, x):
        """ rend la prediction sur x
            x: une description
        """
        res = int(self.labels[np.argmax(self.score(x))])
        return res

    
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        
        self.niter_max = niter_max
        self.w = []
        for i in range(input_dimension):
            self.w.append((np.random.uniform()*2 - 1)*0.001)
        self.allw = [copy.deepcopy(self.w)]
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        max_index = len(desc_set)
        index_tab = []
        for i in range(max_index):
            index_tab.append(i)
        np.random.shuffle(index_tab)
        self.seuil = 0.01
        for i in range(self.niter_max):
            for i in index_tab:
                grad = desc_set[i].T*(np.dot(desc_set[i],self.w) - label_set[i])
                prev_w = copy.deepcopy(self.w)
                self.w = self.w - self.learning_rate*grad
                if(self.history):
                    self.allw.append(copy.deepcopy(self.w))
                temp_vect = np.absolute(prev_w - self.w)
                convergence = 0
                for elt in temp_vect:
                    convergence += elt*elt
                convergence = np.sqrt(convergence)
                if(convergence < self.seuil):
                    break
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if(score > 0):
            res = 1
        elif(score < 0):
            res = -1
        return res
    
    
    
    
    
    import sys 
    
    
    
    
        
    
   
    
    
    
    
    import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        
    """
    sum_p=0
    for i in P:
#         print(len(P))
#         print(i)
        if i!=0 and i!=1:
            
            sum_p+=i*math.log(i)*(-1)
        if i==1:
            sum_p+=0
            
        else:
            sum_p+=0
    return sum_p


    #### A compléter pour répondre à la question posée
    
    
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    val, count = np.unique(Y, return_counts=True)
    tot = sum(count)
    p = []
    for i in count:
        p.append(i/tot)
    return shannon(p)
        
        

    #### A compléter pour répondre à la question posée
    
    
    
    
    
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    val, count = np.unique(Y,return_counts=True)
    return val[np.argmax(count)]
        

# classe_majoritaire(elections_label)
# print(elections_label)
    
    
    
    
    

import sys 

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        ##################
        ## COMPLETER ICI !
        ##################
        liste_entropie = []
        for j in range(len(X[0])):
            xj = X[:,j]
            val, count = np.unique(xj, return_counts=True)
            tot = sum(count)
            e = 0
            for v in range(len(val)):
                e+= (entropie(Y[xj == val[v]])*(count[v]/tot))
            liste_entropie.append(e)
        i_best = np.argmin(liste_entropie)
        gain_max = entropie_classe - liste_entropie[i_best]
        Xbest_valeurs = np.unique(X[:,i_best])
        #############
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud




    
    
    
    
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        ##################
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        return self.racine.classifie(x)
        ##################

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
        
        
        
def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
        output: 
            - un tuple (seuil_trouve, entropie) qui donne le seuil trouvé et l'entropie associée
            - (None , +Inf) si on ne peut pas discrétiser (moins de 2 valeurs d'attribut)
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


# ------------------------ (CORRECTION POUR ENSEIGNANT)
def partitionne(mdesc, mclass, n, s):
    """
    Découpe le dataset en 2 sous dataset à partir des paramètres.
    :param mdesc: Dataset des attributs
    :param mclass: Labels des données
    :param n: numéro de colonne pour laquelle on sépare
    :param s: valeur seuil
    :return: tuple[ndarray, ndarray]
    """
    tmp = np.column_stack((mdesc, mclass))
    return (tmp[mdesc[:,n] <= s][:,:-1], tmp[mdesc[:,n] <= s][:,-1]), (tmp[mdesc[:,n] > s][:,:-1], tmp[mdesc[:,n] > s][:,-1])

# import graphviz as gv

class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe

        if exemple[self.attribut] <= float(self.seuil):
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils['inf'].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            return self.Les_fils['sup'].classifie(exemple)

    
def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g

    
    
def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        liste_entropie = []
        for j in range(len(X[0])):
            xj = X[:,j]
            val, count = np.unique(xj, return_counts=True)
            tot = sum(count)
            e = 0
            for v in range(len(val)):
                e+= (entropie(Y[xj == val[v]])*(count[v]/tot))
            liste_entropie.append(e)
        # i_best = np.argmin(liste_entropie)
        i_best = np.argmax(entropie_classe - np.array(liste_entropie))
        gain_max = entropie_classe - liste_entropie[i_best]
        tmp, liste_vals = discretise(X,Y,i_best)
        # print(i_best)
        Xbest_seuil = tmp[0]
        Xbest_tuple = partitionne(X,Y, i_best,Xbest_seuil)
        # print(liste_entropie)
        ############
        
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud




class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------
