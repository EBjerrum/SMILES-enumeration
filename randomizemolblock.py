import random 
from rdkit.six.moves import range 
from rdkit import Chem

def RandomizeMolBlock(molB): 
	splitB = molB.split('\n') 
	res = [] 
	res.extend(splitB[0:3]) 
	idx = 3 
	inL = splitB[idx] 
	res.append(inL) 
	nAts = int(inL[0:3]) 
	nBonds = int(inL[3:6]) 
   
	idx += 1 
	atLines = splitB[idx:idx + nAts] 
   
	order = list(range(nAts)) 
	random.shuffle(order, random=random.random) 
   
	for i in order: 
		res.append(atLines[i]) 
   
	#print ('ORDER:',order)
	idx += nAts 
	for i in range(nBonds): 
		inL = splitB[idx] 
		idx1 = int(inL[0:3]) - 1 
		idx2 = int(inL[3:6]) - 1 
		idx1 = order.index(idx1) 
		idx2 = order.index(idx2) 
		inL = '% 3d% 3d' % (idx1 + 1, idx2 + 1) + inL[6:] 
		res.append(inL) 
		idx += 1

	#Charges
	for i in range(idx, len(splitB)):
		if splitB[i][0:6] == "M  CHG":
			line = splitB[i]
			chargeline = line.split()
			col = line[0:9]
			for i in range(3,len(chargeline),2):
				col = col +"%4i%4i"%(order.index(int(chargeline[i])-1)+1,int(chargeline[i+1])+1)
			#print (col)
			res.append(col)

	res.append('M  END') 
	return '\n'.join(res) 

