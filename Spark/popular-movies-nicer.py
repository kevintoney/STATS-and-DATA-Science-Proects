from pyspark import SparkConf, SparkContext

def loadMovieNames():
    movieNames = {}
    with open("C:/SparkCourse/ml-100k/u.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames
#load movie names function by creating an empty dictionary,
#opening up the u.item file
#splitting the rdd fields by | pipe bar 
#and extracting the movie id and getting the title. 

conf = SparkConf().setMaster("local").setAppName("PopularMovies")
sc = SparkContext(conf = conf)

nameDict = sc.broadcast(loadMovieNames())
#broadcast variable. 

lines = sc.textFile("file:///SparkCourse/ml-100k/u.data")
movies = lines.map(lambda x: (int(x.split()[1]), 1))
#use a mapper like before. 
movieCounts = movies.reduceByKey(lambda x, y: x + y)

flipped = movieCounts.map( lambda x : (x[1], x[0]))
sortedMovies = flipped.sortByKey()

sortedMoviesWithNames = sortedMovies.map(lambda countMovie : (nameDict.value[countMovie[1]], countMovie[0]))
#map the rdd sortedMovies by having the countMovie variable, with a space colon
#then, calling the nameDict dictionary, using the value function
#to list the 2nd field, and then the first. 

results = sortedMoviesWithNames.collect()

for result in results:
    print (result)
