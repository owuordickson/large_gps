# Mining Large Gradual Patterns 
We implement a chunked layout for loading gradual patterns' binary matrices into memory in order to improve the efficiency of mining large data sets for gradual patterns. The research paper is available via this link:

* Owuor D.O., Laurent A. (2021) Efficiently Mining Large Gradual Patterns Using Chunked Storage Layout. In: Bellatreche L., Dumas M., Karras P., Matulevičius R. (eds) Advances in Databases and Information Systems. ADBIS 2021. Lecture Notes in Computer Science, vol 12843. Springer, Cham. https://doi.org/10.1007/978-3-030-82472-3_4

### Requirements:
You will be required to install the following python dependencies before using <em><strong>ACO</strong>-GRAANK</em> algorithm:<br>
```
                   install python (version => 3.6)

```

```
                    $ pip3 install numpy pandas h5py~=3.3.0 python-dateutil~=2.8.1

```

### Usage:
Use it a command line program with the local package to mine gradual patterns:

```
$python3 src/main.py -a 'aco_ch' -f data/DATASET.csv -u 1
```

where you specify the input parameters as follows:<br>
* <strong>-f</strong>: [required] a file in csv format <br>
* <strong>-u</strong>: [optional] chunk size ```default = 1``` <br>

### License:
* MIT

### References
* Owuor, D.O., Runkler, T., Laurent, A. et al. Ant colony optimization for mining gradual patterns. Int. J. Mach. Learn. & Cyber. (2021). https://doi.org/10.1007/s13042-021-01390-w
* Owuor, D., Laurent, A., Orero, J.: Mining fuzzy-temporal gradual patterns. In: 2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), pp. 1–6. IEEE, New York, June 2019.  https://doi.org/10.1109/FUZZ-IEEE.2019.8858883
* Owuor, D., Laurent, A., Orero, J., Lobry, O.: Gradual pattern mining tool on cloud. In: Extraction et Gestion des Connaissances: Actes EGC’2021 (2021)
Google Scholar
* Owuor, Dickson Odhiambo., Laurent, Anne, Orero, Joseph Onderi: Exploiting IoT data crossings for gradual pattern mining through parallel processing. In: Bellatreche, L., et al. (eds.) TPDL/ADBIS/EDA -2020. CCIS, vol. 1260, pp. 110–121. Springer, Cham (2020).  https://doi.org/10.1007/978-3-030-55814-7_9


