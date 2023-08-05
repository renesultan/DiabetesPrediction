#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 8 2022

@author: renesultan, liamloughman
"""

import pandas as pd 

# Exercice 1

liste_max = []
liste_min = []

def normalize(dataset):
    result = dataset.copy()
    for elements in dataset.columns:
        max_v = dataset[elements].max()
        liste_max.append(max_v)
        min_v = dataset[elements].min()
        liste_min.append(min_v)
        result[elements] = (dataset[elements] - min_v) / (max_v - min_v)
    return result

dataset = pd.read_csv("diabete.csv")
dataset1 = normalize(dataset)

def normalize_input_to_predict(x_pred):
    normalized_input=[]
    for i in range(len(x_pred)):
        valeur = (x_pred[i] - liste_min[i]) / (liste_max[i] - liste_min[i])
        normalized_input.append(valeur)
    return normalized_input

def distance(row,normalized_input):
    dist = abs(dataset1[' Number of times pregnant'][row] - normalized_input[0]) 
    + abs(dataset1[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'][row] - normalized_input[1]) 
    + abs(dataset1[' Diastolic blood pressure (mm Hg)'][row] - normalized_input[2]) 
    + abs(dataset1[' Triceps skin fold thickness (mm)'][row] - normalized_input[3]) 
    + abs(dataset1['2-Hour serum insulin (mu U/ml)'][row] - normalized_input[4]) 
    + abs(dataset1['Body mass index (weight in kg/(height in m)^2)'][row] - normalized_input[5]) 
    + abs(dataset1['Diabetes pedigree function'][row] - normalized_input[6]) 
    + abs(dataset1['Age (years)'][row] - normalized_input[7])
    return(dist)

def d_liste(normalized_input):
    distance_liste=[]
    for i in range (len(dataset)):
        distance_liste = distance_liste + [[distance(i,normalized_input),dataset[" Class variable (0 or 1)"][i]]]
    #Tri a bulle non optimise
    #for k in range (1,len(distance_liste)):
    #    for j in range (1,len(distance_liste)):
    #        if distance_liste[j][0]<distance_liste[j-1][0]:
    #            distance_liste[j-1],distance_liste[j]=distance_liste[j],distance_liste[j-1]
    #tri optimise
    distance_liste.sort(key = lambda distance_liste: distance_liste[0]) 
    return(distance_liste)

def knn(x_pred,k):
    distance_liste = d_liste(normalize_input_to_predict(x_pred))
    liste_knn = distance_liste[0:k]
    zero = 0
    un = 0
    for i in range (len(liste_knn)):
        if liste_knn[i][1] == 0:
            zero = zero + 1
        else:
            un = un+1
    if zero == un:
        k = k + 1
        return(knn(x_pred,k))
    if max(zero,un) == zero:
        return(0)#le patient n'a pas besoin de traitement
    else:
        return(1)#le patient a besoin d'un traitement

new = [1, 89, 67, 24, 80, 23, 0.6, 65]

# Pour determiner la classe du nouveau patient saisir knn(new,10) ou nimporte quel k que vous desirez

# Exercice 2

# Pour les 10 derniers %

BD = {' Number of times pregnant':[],
      ' Plasma glucose concentration a 2 hours in an oral glucose tolerance test':[],
      ' Diastolic blood pressure (mm Hg)':[],
      ' Triceps skin fold thickness (mm)':[],
      '2-Hour serum insulin (mu U/ml)':[],
      'Body mass index (weight in kg/(height in m)^2)':[],
      'Diabetes pedigree function':[],
      'Age (years)':[],
      ' Class variable (0 or 1)':[]}

TEST = {' Number of times pregnant':[],
      ' Plasma glucose concentration a 2 hours in an oral glucose tolerance test':[],
      ' Diastolic blood pressure (mm Hg)':[],
      ' Triceps skin fold thickness (mm)':[],
      '2-Hour serum insulin (mu U/ml)':[],
      'Body mass index (weight in kg/(height in m)^2)':[],
      'Diabetes pedigree function':[],
      'Age (years)':[],
      ' Class variable (0 or 1)':[]}

for i in range(691): #len(dataset)*0.9 = 768*0.9 = 691.2 soit 691
    BD[' Number of times pregnant'].append(dataset1[' Number of times pregnant'][i])
    BD[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'].append(dataset1[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'][i])
    BD[' Diastolic blood pressure (mm Hg)'].append(dataset1[' Diastolic blood pressure (mm Hg)'][i])
    BD[' Triceps skin fold thickness (mm)'].append(dataset1[' Triceps skin fold thickness (mm)'][i])
    BD['2-Hour serum insulin (mu U/ml)'].append(dataset1['2-Hour serum insulin (mu U/ml)'][i])
    BD['Body mass index (weight in kg/(height in m)^2)'].append(dataset1['Body mass index (weight in kg/(height in m)^2)'][i])
    BD['Diabetes pedigree function'].append(dataset1['Diabetes pedigree function'][i])
    BD['Age (years)'].append(dataset1['Age (years)'][i])
    BD[' Class variable (0 or 1)'].append(dataset1[' Class variable (0 or 1)'][i])
    
for i in range (691,768):
    TEST[' Number of times pregnant'].append(dataset1[' Number of times pregnant'][i])
    TEST[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'].append(dataset1[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'][i])
    TEST[' Diastolic blood pressure (mm Hg)'].append(dataset1[' Diastolic blood pressure (mm Hg)'][i])
    TEST[' Triceps skin fold thickness (mm)'].append(dataset1[' Triceps skin fold thickness (mm)'][i])
    TEST['2-Hour serum insulin (mu U/ml)'].append(dataset1['2-Hour serum insulin (mu U/ml)'][i])
    TEST['Body mass index (weight in kg/(height in m)^2)'].append(dataset1['Body mass index (weight in kg/(height in m)^2)'][i])
    TEST['Diabetes pedigree function'].append(dataset1['Diabetes pedigree function'][i])
    TEST['Age (years)'].append(dataset1['Age (years)'][i])
    TEST[' Class variable (0 or 1)'].append(dataset1[' Class variable (0 or 1)'][i])


def distance_2(row_BD,x_pred):
    dist = abs(BD[' Number of times pregnant'][row_BD] - x_pred[0]) 
    + abs(BD[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'][row_BD] - x_pred[1]) 
    + abs(BD[' Diastolic blood pressure (mm Hg)'][row_BD] - x_pred[2]) 
    + abs(BD[' Triceps skin fold thickness (mm)'][row_BD] - x_pred[3]) 
    + abs(BD['2-Hour serum insulin (mu U/ml)'][row_BD] - x_pred[4]) 
    + abs(BD['Body mass index (weight in kg/(height in m)^2)'][row_BD] - x_pred[5]) 
    + abs(BD['Diabetes pedigree function'][row_BD] - x_pred[6]) 
    + abs(BD['Age (years)'][row_BD] - x_pred[7])
    return(dist)

def d_liste_2(x_pred):
    distance_liste=[]
    for i in range (691):
        distance_liste = distance_liste + [[distance_2(i,x_pred),BD[" Class variable (0 or 1)"][i]]]
    distance_liste.sort(key = lambda distance_liste: distance_liste[0]) 
    return(distance_liste)

def knn_2(x_pred,k):
    distance_liste = d_liste_2(x_pred)
    liste_knn = distance_liste[0:k]
    zero = 0
    un = 0
    for i in range (len(liste_knn)):
        if liste_knn[i][1] == 0:
            zero = zero + 1
        else:
            un = un+1
    if zero == un:
        k = k + 1
        return(knn_2(x_pred,k))
    if max(zero,un) == zero:
        return(0)#le patient n'a pas besoin de traitement
    else:
        return(1)#le patient a besoin d'un traitement
    
def fiabilite(k):
    valide=0
    for i in range(77):
        x_pred=[]
        x_pred.append(TEST[' Number of times pregnant'][i])
        x_pred.append(TEST[' Plasma glucose concentration a 2 hours in an oral glucose tolerance test'][i])
        x_pred.append(TEST[' Diastolic blood pressure (mm Hg)'][i])
        x_pred.append(TEST[' Triceps skin fold thickness (mm)'][i])
        x_pred.append(TEST['2-Hour serum insulin (mu U/ml)'][i])
        x_pred.append(TEST['Body mass index (weight in kg/(height in m)^2)'][i])
        x_pred.append(TEST['Diabetes pedigree function'][i])
        x_pred.append(TEST['Age (years)'][i])
        x_pred.append(TEST[' Class variable (0 or 1)'][i])
        new_class = knn_2(x_pred,k)
        if new_class == x_pred[8]:
            valide = valide + 1
    pourcentage = (valide*100)/77
    return(pourcentage)

# Pour tester la fiabiliter il faut enlever les # du code ci dessous
#Ce code prend du temps
liste = []
for k in range (1,692):
    liste.append([k,fiabilite(k)])
liste.sort(key = lambda liste: liste[1])
print(liste)

 