import numpy
from matplotlib import pyplot
import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
from data_classes import Level, Cluster

'''
test = {frozenset((1,2)):3.4, frozenset((2,3)):4.5}

test.pop(frozenset((1,2)))
#print(test)

for ids in test:
    id1, id2 = ids
    #print(f'{id1} and {id2}')

list1 = ['hello', 'one', 'tow', 'one']
dict1 = {'hello':1, 'two':2}
for word in list1:
    if word not in dict1:
        dict1[word] = 1
    else:
        dict1[word] += 1

s = " hello my name is colin. I like chocolate milk"

print(s[:3])
print(len(s))
'''

all_clusters = numpy.arange(0,10)
merged_clusters = [2,4]
new_cluster = 10