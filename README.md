# Online evolutionary Spiking Neural Network unsupervised anomaly detector

## Description

 Python implementation of OeSNN-UAD model. Model finds anomalies in one dimensional data streams. Theoretical basics about this model could be find here: https://arxiv.org/pdf/1912.08785.pdf.

## Dependencies

* numpy
* PyEMD (pip install EMD-signal) 

## Package instalation

To install package for python you should type in terminal:

```bash
    pip install OeSNN-AD
```

## Usage

Our model require from data stream to be numpy array. Additional model parameters are passed as arguments in object constructor.

The following code snippet shows package basic usage.

```python
    from oesnn_ad import OeSNNAD
    import numpy as np

    data_stream = np.array([1, 2, 3, 4, 5])
    model = oesnn_ad(data_stream)

    results = model.predict()
```

## Parameters

The following table shows model parameters and their values range.

<center>

| Parameter       | Default value | Minimal value | Maximum value |
| --------------- | :-----------: | :-----------: | :-----------: |
| window_size     |      100      |       1       |     -         |
| num_in_neurons  |      10       |       1       |     -         |
| num_out_neurons |      50       |       1       |     -         |
| ts_factor       |      1000     |       0       |     -         |
| mod             |      0.6      |       0       |     1         |
| c_factor        |      0.6      |       0       |     1         |
| epsilon         |      2        |       2       |     -         |
| ksi             |      0.9      |       0       |     1         |
| sim             |      0.15     |       0       |     -         |
| beta            |      1.6      |       0       |     -         |

</center>