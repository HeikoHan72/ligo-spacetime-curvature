# Data

```
data/raw/
    gw150914/
        H-H1_LOSC_4_V2-1126259446-32.hdf5     Hanford strain, 4096 Hz, 32 s
        L-L1_LOSC_4_V2-1126259446-32.hdf5     Livingston strain, 4096 Hz, 32 s
    trainingset_v1d1_metadata.csv             Gravity Spy glitch metadata
```

## GW150914 strain

Source: https://gwosc.org/events/GW150914/ (dataset DOI 10.7935/K5MW2F23).
Download the 4096 Hz, 32 s, HDF5 files for H1 and L1. Each file is about 1 MB
and is centred on GPS 1126259462. Any file name containing `H1` or `L1` and the
extension `.hdf5` is found automatically.

## Gravity Spy metadata

7,966 glitch events in 22 classes, recorded at H1 and L1 between September 2015
and January 2017. Citation: Zevin et al. (2017), Classical and Quantum Gravity
34, 064003.

## Terms of use

Check the terms of use of both data sets on their original pages before
redistributing the files in a public repository. If they do not allow it,
remove the files from `data/raw/` and list them in `.gitignore`; the code reads
them from the paths above.
