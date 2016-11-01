import sys, re, os, codecs
from optparse import OptionParser

try:
    import ujson as json
except ImportError:
    import json
sys.path.insert(0, ".")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.', 'gflparser'))
import gfl_parser
import json
import copy

def CreateLeftBranchingMultiword(mw):
    toks = mw.split("_")
    for tokInd in range(len(toks)-1):
        yield str(tokInd+1)+" -> "+str(tokInd+2)

def CreateRightBranchingMultiword(mw):
    toks = mw.split("_")
    for tokInd in range(len(toks)-1):
        yield str(tokInd+2)+" -> "+str(tokInd+1)

# Returns 'whiteList' and 'blackList', which contain lists of arcs to either include or exclude.
# If an arc appears in neither, the annotation does not specify its inclusion or exclusion.
def getArcLists(filename,leftMulti=False,uneven=False):
    whiteLists = list()
    blackLists = list()
    rf = open(filename)
    for line in rf:
        whiteList = list()
        blackList = list()
        brackets  = list()
        jsonobj = json.loads(line)
        sentenceLength = len(jsonobj["sent"].split())
        parse = gfl_parser.parse(jsonobj["sent"].split(), jsonobj["anno"].replace("\r\n", "\n").replace("\n\n","\n").replace("[","(").replace("]",")"), check_semantics=True)
        parseJ = parse.to_json()
        feCoverage = {}
        feSolidCover = {}

        for dep in [dep for dep in parseJ['deps'] if dep[2] == 'fe']:
            if dep[0] not in feCoverage.keys():
                feCoverage[dep[0]] = []

            if dep[1][0:2] != "FE":
                feCoverage[dep[0]].append(parseJ['tokens'].index(dep[1][2:-1])+1)
            else:
                feCoverage[dep[0]].append(dep[1])

        for k,v in feCoverage.iteritems():
            if len([item for item in v if isinstance(item,str)]) == 0:
                feSolidCover[k] = [sorted(v)[0],sorted(v)[-1]]

        while len(feSolidCover.keys()) != len(feCoverage.keys()):
            # Replace strings with limits if possible
            for k,v in feCoverage.iteritems():
                for solidKey in feSolidCover.keys():
                    if solidKey in v:
                        i = v.index(solidKey)
                        v[i:i+1] = feSolidCover[solidKey][0], feSolidCover[solidKey][1]
                feCoverage[k] = v

            # Put completed FE nodes into Solid
            for k,v in feCoverage.iteritems():
                if len([item for item in v if isinstance(item,str)]) == 0:
                    feSolidCover[k] = [sorted(v)[0],sorted(v)[-1]]

        for k,v in feSolidCover.iteritems():
            brackets.append(v)

        # Handle Multiword Expressions
        for node in parseJ["nodes"]:
            if node[0:2] == "MW":
                if leftMulti:
                    gen = CreateLeftBranchingMultiword(node[3:-1])
                else:
                    gen = CreateRightBranchingMultiword(node[3:-1])
                for dep in gen:
                    whiteList.append(dep.strip())

        # Handle Coordination Nodes
        for coord in parseJ["coords"]:
            if uneven:
                if len(coord[2]) == 1 and coord[2][0][0] == "W" and len(coord[1]) == 2:
                    whiteList.append(str(parseJ['tokens'].index(coord[1][0][2:-1])+1)+" -> "+str(parseJ['tokens'].index(coord[2][0][2:-1])+1))
                    whiteList.append(str(parseJ['tokens'].index(coord[2][0][2:-1])+1)+" -> "+str(parseJ['tokens'].index(coord[1][1][2:-1])+1))
            else:
                if len(coord[2]) == 1 and coord[2][0][0] == "W":
                    for target in coord[1]:
                        if target[0] == "W":
                            whiteList.append(str(parseJ['tokens'].index(coord[2][0][2:-1])+1) + " -> " + str(parseJ['tokens'].index(target[2:-1])+1))
        deps = parseJ["deps"]
    

        for dep in deps:
            if dep[0][0] == "W" and dep[1][0] == "W":
                whiteList.append(str(parseJ['tokens'].index(dep[0][2:-1])+1) + " -> " + str(parseJ['tokens'].index(dep[1][2:-1])+1))
            elif dep[0][0:2] == "MW":
                whiteList.append(str(parseJ['tokens'].index(dep[0][3:-1].split("_")[-1])+1)+ " -> " + str(parseJ['tokens'].index(dep[1][2:-1])+1))
            elif dep[1][0:2] == "MW":
                whiteList.append(str(parseJ['tokens'].index(dep[0][2:-1])+1)+ " -> " + str(parseJ['tokens'].index(dep[1][3:-1].split("_")[-1])+1))

        # Add reversed whitelist (Only really useful if we are not doing 'absolute' white-listing, but doesn't hurt in any case)
        #for dep in whiteList:
        #    deps = dep.strip().split(" -> ")
        #    blackList.append(deps[1]+" -> "+deps[0])

        # Add 'No External Children' and 'Single External Head' constraints
        for bracket in brackets:
            b_limit_l = bracket[0]
            b_limit_r = bracket[1]
            b_hasHead = True # FIXME: Determine whether the head of the bracket is known.
                              # If it is, we need to blacklist any arcs with heads external to the bracket.
                              # If it isn't, still do the blacklisting, except leave external heads to the bracket head as neutral.
            
            for headIndex in range(b_limit_l,b_limit_r):
                for childIndex in range(0,sentenceLength):
                    if ((childIndex < b_limit_l) or childIndex > b_limit_r):
                        # blacklist external children
                        blackList.append(str(headIndex)+" -> "+str(childIndex))
                    
                        # blacklist external heads (notation reversed)
                        blackList.append(str(childIndex)+" -> "+str(headIndex))
                        
            
            if b_hasHead:
                head = b_limit_l
                # FIXME: Unblacklist external connections to the bracket head.
                for headIndex in range(0,sentenceLength):
                    if ((headIndex < b_limit_l) or headIndex > b_limit_r):
                        try:
                            blackList.remove(str(headIndex)+" -> "+str(head))
                        except ValueError:
                            pass
                pass

        # Clean up
        blackList = [dep for dep in blackList if dep not in whiteList]

        whiteLists.append(whiteList)
        blackLists.append(blackList)
    rf.close()
    
        
    return whiteLists, blackLists