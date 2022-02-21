## Setup
Install the requirements with `pip install -r requirements.txt`.

## Experiments
To run the MNIST experiment (Figure 6 in the paper):
```bash
$ python run_mnist.py
```

To run the tensor decomposition experiments including the baselines (Figure 4 in the paper):

```bash
$ python run_decomposition.py --target_type=tucker --target_dims="(7,7,7,7,7)" --target_rank="[2,3,4,3,2]"  --nruns=50 --result_pickle=results-all-tucker.pickle

$ python run_decomposition.py --target_type=triangle --target_dims="(7,7,7,7,7)" --target_rank="[5,2,5,2,2]" --nruns=50 --result_pickle=results-all-triangle.pickle

$ python run_decomposition.py --target_type=tt --target_dims="(7,7,7,7,7)" --target_rank="[2,3,6,5]" --nruns=50 --result_pickle=results-all-tt.pickle

$ python run_decomposition.py --target_type=tr --target_dims="(7,7,7,7,7)" --target_rank="[2,3,4,5,5]" --nruns=50 --result_pickle=results-all-tr.pickle
```

To run the image completion experiment (Figure 5 in the paper):

```bash
$ python run_completion.py --target_file=data/Einstein.mat --epochs 500 --method greedy_ALS
```