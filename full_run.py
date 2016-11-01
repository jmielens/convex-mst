import sys, string, random, os, re, subprocess
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-t","--train", dest="trainConll", help="Training Conll")
parser.add_option("-g","--gfl", dest="trainGfl", help="Training GFL")
parser.add_option("-e","--test", dest="testConll", help="Testing Conll")
parser.add_option("-o","--output", dest="output", help="Output File", default="out")
parser.add_option("-i","--iterations", dest="iter", help="Number of Iterations", default="100")
parser.add_option("-l","--length", dest="length", help="Max number of Tokens", default="20")
(options, args) = parser.parse_args()


subprocess.call(['python','full_arcs_sparse_gfl.py', '-f', options.testConll, '-l', options.length, '-o', 'test'])
subprocess.call(['python','full_arcs_sparse_gfl.py', '-f', options.trainConll, '-g', options.trainGfl, '-o', 'gfl'])


subprocess.call(['python', 'test_exp.py', '-q','-f', options.trainConll, '-t', options.testConll, '-c', options.length, '-o', options.output, '-e', options.iter, '-g', '5.0', '-u', '0.0'])
