#import the libraries we need to scrape a webpage
import urllib3
import requests
from bs4 import BeautifulSoup

# specify the url
input_page = 'http://www.espn.com/college-football/playbyplay?gameId=400935256'

#query the website and return the html to the variable page
#http = urllib3.PoolManager()
res = requests.get(input_page)

#page = http.request('GET', input_page)

#parse the html using beautiful soup and store in varaible soup
#soup = BeautifulSoup(page, 'html.parser')
soup_test = BeautifulSoup(res.text, 'html.parser')

#take out the <div> of the play's text and get its value.

name_box_test = soup_test.find_all('h3')
play = soup_test.find_all('span', attrs={'class':'post-play'})

#name = name_box_test.text.strip() # strip removes the starting
#and trailing carrots. 
#play = play.text.strip()
output = open("C:/Users/kevin/Desktop/cal_scrapping.txt", "w+")

for i in range(1, 9):
    print(name_box_test[i].text + " " + play[i].text, file=output)
    
