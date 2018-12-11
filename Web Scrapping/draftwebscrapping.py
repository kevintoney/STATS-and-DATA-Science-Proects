#import the libraries we need to scrape a webpage
import urllib3
import requests
from bs4 import BeautifulSoup
import numpy as np
import re

# specify the url
input_start = 'https://www.pro-football-reference.com/years/'
year = list(range(1971, 2014))
input_end = '/draft.htm'

#query the website and return the html to the variable page
#http = urllib3.PoolManager()
urls = []
for i in range(0, 43):
    urls.append(input_start + str(year[i]) + input_end)

url = urls[0]
html = requests.get(url)
soup = BeautifulSoup(html.text, 'html.parser')
drafts = []


for i in soup.find_all(['th','td', '"></td>']):
    if str(i.find(string=True))[0]:
        drafts.append(str(i.find(string=True)))
    elif '"></td>' in str(i):
        drafts.append('"></td>')
    
             
header = drafts[11:39]
len(header)
print(header)
header = np.array(header)
print(header.shape)
info = list(drafts[39:])
type(info)
len(info)
infoA = np.array(info)
print(infoA.shape)
infoArray = np.reshape(infoA, (458, 28))
print(infoArray[0,])


#make a master array
drafthist = np.vstack((header, infoArray))
print(drafthist.shape)

#add year to array
yearcol = np.array("1971")
yearcol = np.repeat(yearcol, [459], axis=0)
drafthist = np.column_stack((yearcol, drafthist))
print(drafthist.shape)
print(drafthist[:3,])

'''
#now, get data for the rest of the years
for y in range(1,43):
    url = urls[30] 
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    drafts = []
    for i in soup.find_all(['th','td', '"></td>']):
        if str(i.find(string=True))[0]:
            drafts.append(str(i.find(string=True)))
        elif '"></td>' in str(i):
            drafts.append('"></td>')
    
    header = drafts[11:39]
    len(header)
    header = np.array(header)
    print(header.shape)
    info = list(drafts[39:])
    infoA = np.array(info)
    infoArray = np.reshape(infoA, (458, 28))
    print(infoArray[0,])
    #make a master array
    drafthist = np.vstack((header, infoArray))
    print(drafthist.shape)
    #add year to array
    yearcol = np.array(year)
    yearcol = np.repeat(yearcol, [459], axis=0)
    drafthist = np.column_stack((yearcol, drafthist))


output = open("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/draft_history.txt", "w+")

print(drafthist, file=output)
'''
