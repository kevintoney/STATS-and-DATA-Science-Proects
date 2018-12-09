import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crimes = pd.read_csv("Desktop/Career Prep/Kevin work Portfolio/Personal Project/Crimes_-_2001_to_present.csv")

crimes.dtypes
crimes.head(5)

#sort by primary type
crimes = crimes.sort_values(by='Primary Type')
crimes.head(3)

#amount of FBI Codes per primary type. 
crimes.pivot_table(index='Primary Type', columns='FBI Code', values = 'Ward', aggfunc='count')
crimes.pivot_table(index='Primary Type', columns='IUCR', values = 'Ward', aggfunc='count')
#IUCR's are 4 digit crime codes in the state of Illinois. 

######
#my favorite pivot tables. 
crimes.pivot_table(index='Primary Type', columns='Domestic', values = 'Ward', aggfunc='count')
#crimes in many different categories are domestic.
#maybe, I should visualize where domestic crimes happen. 
#or, I can predict whether these crimes are domestic or not. 
#there seems to be less domestic crimes than not. 
counts = crimes.pivot_table(index='Block', columns='Domestic', values = 'Ward', aggfunc='count')
block_counts = crimes.groupby(['Block', 'Domestic'])['FBI Code'].count()
nodomestic = block_counts[0:100729:2]
nodomestic = nodomestic[:5]
domestic = block_counts[1:100729:2]
domestic = domestic[:5]
#maybe, I should visualize where domestic crimes happen. 
######

#this district pivot table isn't that useful. 
crimes.pivot_table(index='District', columns='Domestic', values = 'Ward', aggfunc='count')

crimes.describe()
#for the small dataset
#range of X coordinate is 1,100,317 to 1,203,152
#range of y coordinate is 1,813,910 to 1,951,493

