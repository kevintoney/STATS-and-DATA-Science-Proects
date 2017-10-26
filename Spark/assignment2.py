from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("TotalCustomerSpend")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    cust_id = int(fields[0])
    money_spent = float(fields[2])
    return (cust_id, money_spent)

spenddata = sc.textFile("file:///SparkCourse/customer-orders.csv")
customerspendpair = spenddata.map(parseLine)

dollarsspent = customerspendpair.reduceByKey(lambda x, y: x+y)

dollarsspent_prep = dollarsspent.map(lambda x: (x[1], x[0]))
moneyspent_sorted = dollarsspent_prep.sortByKey()
moneyspent_bycustomer = moneyspent_sorted.collect()

for customer in moneyspent_bycustomer:
    print("{:.2f}".format(customer[0]), customer[1])
    #use this format to round numbers to two decimal places. 