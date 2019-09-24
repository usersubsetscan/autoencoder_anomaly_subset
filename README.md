## Readme

Add project directory to PYTHONPATH

```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Usage:

Scanning and getting scores from clean and anomalous data sets

```

usage: main.py [-h] [--customfunction CUSTOMFUNCTION]
               [--clean_ssize CLEAN_SSIZE] [--anom_ssize ANOM_SSIZE]
               [--bgddata BGDDATA] [--bgdlabels BGDLABELS]
               [--cleandata CLEANDATA] [--anomdata ANOMDATA] [--model MODEL]
               [--modelclass MODELCLASS] [--conditional CONDITIONAL]
               [--constraint CONSTRAINT] [--scorefunc SCOREFUNC]
               [--pvaltest PVALTEST] [--layers LAYERS [LAYERS ...]]
               [--run RUN] [--restarts RESTARTS] [--resultsfile RESULTSFILE]

optional arguments:
  -h, --help            show this help message and exit
  --customfunction CUSTOMFUNCTION
                        custom function after extraction of activation
                        (experimental)
  --clean_ssize CLEAN_SSIZE
                        sample size of clean records for evaluation
  --anom_ssize ANOM_SSIZE
                        sample size of anomalous records for evaluation
  --bgddata BGDDATA     background records
  --bgdlabels BGDLABELS
                        background labels
  --cleandata CLEANDATA
                        clean records
  --anomdata ANOMDATA   anom records
  --model MODEL         path to the model
  --modelclass MODELCLASS
                        path to model class
  --conditional CONDITIONAL
                        whether or not to compute pvalues ranges conditioned
                        on each class label
  --constraint CONSTRAINT
                        search group
  --scorefunc SCOREFUNC
                        scoring function
  --pvaltest PVALTEST   type of test
  --layers LAYERS [LAYERS ...]
                        name or index of layer(s) to extract
  --run RUN             number of times to sample and run scan
  --restarts RESTARTS   number of times to perform iterative restart
  --resultsfile RESULTSFILE
                        output file containing results
```

Visualizing detection power:

```
usage: detectionpower.py [-h] [--cleanscores CLEANSCORES]
                         [--anomscores ANOMSCORES]

optional arguments:
  -h, --help            show this help message and exit
  --cleanscores CLEANSCORES
                        clean scores file path
  --anomscores ANOMSCORES
                        anomalous scores file path
```


Results file format

The following results are written to a textfile in the following format.

`score precision recall len_image_sub len_node_sub optimal_alpha node_subs image_subs`

The delimeter is a single whitespace

`node_subs` is a `,` seperated list of identified anomalous nodes

Example on detecting BIM noise on a resnet model in the [examples](examples/cifar10_adversarial.py) folder