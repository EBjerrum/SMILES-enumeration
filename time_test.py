import timeit
setup = """from SmilesEnumerator import SmilesEnumerator, SmilesIterator\n
import numpy as np
sm_en = SmilesEnumerator()\n
smiles = np.array([ "CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2","CCC[S@@](=O)c1ccc2c(c1)[nH]/c(=N/C(=O)OC)/[nH]2"]*100)\n
sm_en.fit(smiles, extra_chars=["\\\\"])\n
sm_it = SmilesIterator(smiles, np.array([1,2]*100), sm_en, batch_size=128, shuffle=True)\n
"""
command = """X,y = sm_it.next()"""

print timeit.timeit(command, number=100, setup=setup)/100
#Approximate 43ms pr. 128


