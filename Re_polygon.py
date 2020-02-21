import os
import re
import numpy as np
import pandas as pd
import h5py
import json

def loadFile(folder):

	filename = folder + '/Paleotopo_P100.gmt'
	# f = open(folder+'/Paleotopo_P100.gmt')	
	# lines = f.readlines()
	
	# with open(filename) as f:
	# 	lines = f.read().splitlines() 

	with open(filename) as f:
		lines = [line.rstrip() for line in f]
	
	print('Lines : ',len(lines))
	# return f
	return lines

def replaceValues():

	lines = loadFile('Init_topo_polygon')
	pattern = r'(?<=\|)\d+'
	
	new_list = []
	for l in lines:
		new_list.append(re.sub(pattern,'1000',l))

	outF = open("myOutFile.txt", "w")
	for line in new_list:
		# write line to output file
		outF.write(line)
		outF.write("\n")
	outF.close()
	# with open('ReplacedFile.txt','w') as file_out:
	# 	json.dump(new_list,file_out)

	return

def main():

	replaceValues()

if __name__ == "__main__": main()