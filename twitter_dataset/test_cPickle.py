import cPickle


one = [1,1,1]
two = [2,2,2]
three = [3,3,3]

file = open("test.pkl", 'wb')
cPickle.dump((one, two, three), file, protocol=cPickle.HIGHEST_PROTOCOL)
file.close()

file = open("test.pkl", 'rb')
fetch_one, fetch_two, fetch_three = cPickle.load(file)
file.close()

print "fetch_one:", fetch_one
print "fetch_two:", fetch_two
print "fetch_three:", fetch_three

