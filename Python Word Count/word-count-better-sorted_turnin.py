import re
import sys
from pyspark import SparkConf, SparkContext
from operator import add

def normalizeWords(text):
    return re.compile(r'\d+|\W+', re.UNICODE).split(text.lower())
    # the regex r'\d+\W+' gets rid of digits in strings and punctuation marks
    # and compiles all words together no matter the punctuation mark they have. 
    
conf = SparkConf().setMaster("local").setAppName("WordCount_Bible")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///STAT420_HW/bible.txt")
words = input.flatMap(normalizeWords)

wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
#don't use countByValue. 
#create a word mapper by creaeting a tuple which has a 1 value appended
#and then sum the 1 values. 
wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey(ascending=False)
#we will now flip the rdd around to say first the 
#number of times a word appears, and then the words. 
#the number of occurances is now the key we will sort by. 
results = wordCountsSorted.collect()

output = open('bible_results.txt', 'a')

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count, file=output)
        


'''with open('C:\\Users\\kevin\\Desktop\\Fall 2017\\STAT 420\\Homework\\bible_results.txt', 'w') as outfile:
    for result in results:
        count = str(result[0])
        word = result[1].encode('ascii', 'ignore')
        if (word):
            print(word.decode() + ":\t\t" + count)
'''
