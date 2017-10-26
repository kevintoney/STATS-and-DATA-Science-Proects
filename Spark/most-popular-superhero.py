from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("PopularHero")
sc = SparkContext(conf = conf)

def countCoOccurences(line):
    elements = line.split() #split by using whitespace
    return (int(elements[0]), len(elements) - 1)
    #return the hero id, and the number of coocurrances, which is the number of 
    #elements minus the first field. 

def parseNames(line):
    fields = line.split('\"') #split by using \"
    return (int(fields[0]), fields[1].encode("utf8")) #get the name. 

#first action. 
names = sc.textFile("file:///SparkCourse/marvel-names.txt")
namesRdd = names.map(parseNames)

lines = sc.textFile("file:///SparkCourse/marvel-graph.txt")

pairings = lines.map(countCoOccurences)
totalFriendsByCharacter = pairings.reduceByKey(lambda x, y : x + y)
#add values together. 
#the values are the number of cooccurances. 
flipped = totalFriendsByCharacter.map(lambda xy : (xy[1], xy[0]))
#flip the rdd so the #of corrcurances are the keys
#and the values are the super hero's name. 

mostPopular = flipped.max()
#max key value. 

mostPopularName = namesRdd.lookup(mostPopular[1])[0]

print(str(mostPopularName) + " is the most popular superhero, with " + \
    str(mostPopular[0]) + " co-appearances.")
