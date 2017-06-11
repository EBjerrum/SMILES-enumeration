# SMILES-enumeration
SMILES enumeration for QSAR modelling using LSTM recurrent neural networks


## Example usage
See the if __name__ == "__main__": example in SMILES_enumeration.py

to use in own scripts

```python
from SMILES_enumeration import get_mol_set

#Split into canonical and non-canocal
canonical, s = get_mol_set('CCCC', tries=50)

#All forms found in same set
s = get_mol_set('CCCC', tries=50, split=False)
```

Please note that it currently strips all stereo information.


## Bibliography

Please cite: [SMILES enumeration as Data Augmentation for Network Modeling of Molecules](https://arxiv.org/abs/1703.07076)

```bibtex
@article{DBLP:journals/corr/Bjerrum17,
  author    = {Esben Jannik Bjerrum},
  title     = {{SMILES} Enumeration as Data Augmentation for Neural Network Modeling
               of Molecules},
  journal   = {CoRR},
  volume    = {abs/1703.07076},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.07076},
  timestamp = {Wed, 07 Jun 2017 14:40:38 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/Bjerrum17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```


