from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',') #split lines by commas first. 
    stationID = fields[0] #take the first field. 
    entryType = fields[2] #take the third field. 
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0 #fourth field
    #is a floating numeric field we can do math with. 
    return (stationID, entryType, temperature)

lines = sc.textFile("file:///SparkCourse/1800.csv")
parsedLines = lines.map(parseLine)
#create an rdd. 
maxTemps = parsedLines.filter(lambda x: "TMAX" in x[1])
#filter to find "TMAX"
stationTemps = maxTemps.map(lambda x: (x[0], x[2]))
#this is a tuple. 
maxTemps = stationTemps.reduceByKey(lambda x, y: max(x,y))
#find the minimum value of x and y. minTemps is a new rdd. 
results = maxTemps.collect();

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
    #format the results to be two decimal points. 
    #desired format in quotation marks, then .format(rdd[])
    #immediately following. 
