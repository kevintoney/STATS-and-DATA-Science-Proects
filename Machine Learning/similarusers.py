import sys
from operator import add
from pyspark import SparkContext


def firstMap(line):
	line = line.strip().split('\t')
	user_id = line[0]
	movie_id = line[1] + '-' + line[2]
	print >> sys.stderr, movie_id
	return (movie_id, user_id)

def firstReduce(x,y):
	if type(x) == list and type(y) == list:
		return x + y 
	elif type(x) == list:
		x.append(y)
		return x
	elif type(y) == list:
		y.append(x)
		return y
	else:
		return [x] + [y]

def secondMap(line):
	movie = line[0]
	print >> sys.stderr, "LINE:"
	print >> sys.stderr, line
	users = list(line[1])
	user_TupList = list()
	if u'1488844' in users:
		for u in users:
			user_TupList.append((u, 1))
		print user_TupList
		return user_TupList
	else:
		return [(0,0)]
	
if __name__ == "__main__":
	#if len(sys.argv) != 3:
		#print("Usage: wordcount_v2 <input> <output>", file=sys.stderr)
		#exit(-1)
	sc = SparkContext(appName="WordCountAgain")

	lines = sc.textFile(sys.argv[1], 1)
	movieList = lines.map(firstMap) \
			.reduceByKey(firstReduce) \
			.flatMap(secondMap) \
			.reduceByKey(add) \
			.sortBy(lambda a: a[1])
	
	movieList.saveAsTextFile(sys.argv[2])
	sc.stop()