#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, string, random, os, numpy, re#, libhfst
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f","--file", dest="file", help="Oracle to split")
parser.add_option("-v","--verbose", dest="verbose", help="Print lots of stuff")
parser.add_option("-o","--output", dest="output", help="Output File", default="out")
(options, args) = parser.parse_args()

sentences = open(options.file).read().split("\n\n")
f = open(options.output,'w')
for sentence in sentences:
	heads = {}

	deleted = []
	corrections = []
	lines = sentence.split("\n")
	for line in lines:
		tokens = line.split("\t")
		#print tokens
		if len(tokens) < 3:
			continue
		if tokens[3] == "." or tokens[7] == "p":
			deleted.append(int(tokens[0]))
		else:
			heads[int(tokens[0])] = int(tokens[6])
		corrections.append(len(deleted))
	
	for (k,v) in heads.iteritems():
		tokens = lines[k-1].split("\t")
		tokens[0] = str(int(tokens[0])-corrections[k-1])
		if v == 0:
			tokens[6] = str(v)
		else:
		    tokens[6] = str(int(tokens[6])-corrections[int(tokens[6])-1])
		f.write("\t".join(tokens)+"\n")
		#(k, corrections[k-1])
	f.write("\n")
f.close()