# Abstract Argumentation Solving Graph Dataset

This repository contains a dataset of argumentation frameworks that includes enumerations of sets of extensions for each framework.


## Argumentation Frameworks

An *argumentation framework* is a pair `AF = (A, R)`, where `A` is a set of *arguments* and `R` is a binary relation over `A` that represents the *attack* relation. 

A set `S ⊆ A` is *conflict-free* iff there are no attacking argument in `S`, and an argument `a` is defended by `S` iff any argument that attacks `a` is attacked by an argument in `S`. 
A set `S` is *admissible* iff it is conflict-free and defends all its arguments.
Then, for a set `E ⊆ A`:

1. `E` is a *complete extension* (denoted `CO`) iff it is admissible any argument defended by `E` is in `E`,
2. `E` is the single, unique *grounded extension* (denoted `GR`) iff it is the smallest complete extension (w.r.t. set inclusion),
3. `E` is a *preferred extension* (denoted `PR`) iff it is a maximal complete extension (w.r.t. set inclusion),
4. `E` is a *stable extension* (denoted `ST`) iff it is a complete extension that attacks all arguments not in `E`,
5. `E` is a *semi-stable extension* (denoted `SST`) iff it is a complete extension that maximises the union of its members and the arguments its members attack, and
6. `E` is a *stage extension* (denoted `STG`) iff it is a complete extension that minimises the union of its members and the arguments its members attack.

Denote the set of all `σ`-extensions (where `σ` is one of the above semantics) as `E_σ`.
An argument is *credulously acceptable* w.r.t. `AF` and `σ` iff it is in at least one `σ`-extension `E ∈ E_σ`, and it is *sceptically acceptable* iff it is in all `σ`-extensions `E ∈ E_σ`.


## Dataset Details

The data is derived from the [International Competition on Computational Models of Argumentation](http://argumentationcompetition.org/2017/) from the ICCMA 2017 benchmark datasets.

### Solving Methodology
Each framework was read in `.tgf` format and translated into an `.apx` format using the [`store_af.py`](store_af.py) script.
The `.apx` format is a simple text-based format that lists the arguments and attacks in the framework for input into an answer set programming solver.
The `.apx` files were then solved using the [`ASPARTIX`](https://www.dbai.tuwien.ac.at/research/argumentation/aspartix/) implementations of argumentation solving and the `clingo` answer-set programming solver, using the [`single_solve.py`](single_solve.py) script.



### Data Format
Each framework is stored in Python dictionary with the following keys, this is based on the `pytorch_geometric` data format.

- `x`: node feature matrix with shape `(|A|, 1)` consisting of unique integer identifiers for each argument (type `torch.float32`).
- `edge_index`: the relation `R` encoded in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape [2, num_edges] and type `torch.long`.
- `extensions`: a dictionary mapping keys `σ ∈ {"CO", "GR", "PR", "ST", "SST", "STG"}` to binary tensors with shape `(|E_σ|, |A|)` and type `torch.long`.


For example:

```python
{
    "x": torch.tensor([[0], [1], [2], [3]]),
    "edge_index": torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]]),
    "extensions": {
        "GR": torch.tensor([[1, 0, 1, 0]]),
        # ...
    }
}
```

## Acknowledgements and License

The code is authored by Alex Jackson and is licensed under the [MIT License](LICENSE).

> This work was supported by UK Research and Innovation [grant number EP/S023356/1], in the UKRI Centre for Doctoral Training in Safe and Trusted Artificial Intelligence ([www.safeandtrustedai.org](https:/www.safeandtrustedai.org)).

In addition:

- The source data was provided as part of the [International Competition on Computational Models of Argumentation](http://argumentationcompetition.org/2017/).
- The `ASPARTIX` implementations of argumentation solving were created by the [Database and Artificial Intelligence Group](https://www.dbai.tuwien.ac.at/) at the [Vienna University of Technology](https://www.tuwien.at/).
- The `clingo` answer-set programming solver was created by the [Potassco](https://potassco.org/) project.




