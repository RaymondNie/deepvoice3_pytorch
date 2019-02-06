import os 
import sys

dirname = sys.argv[1]
str = ""

for file in os.listdir(dirname):
    str += "{}/{}/alignment.json,".format(dirname, file)

str = '"' + str[0:-1] + '"'
print(str)
