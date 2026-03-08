# box_solver

Python code for modelling quantum transport setups involving Majorana box qubits.

The project allows the construction of systems with an arbitrary number of Majorana modes connected to multiple electronic leads. Transport properties are computed numerically using quantum master equations via the `qmeq` library.

The code was developed during my PhD research to explore transport behaviour in different Majorana box configurations.

## Model

The framework represents quantum transport systems consisting of

- a Majorana box hosting multiple Majorana modes
- several electronic leads coupled to the box
- tunneling amplitudes and interaction parameters defining the system

Different system geometries can be explored by specifying the number of Majoranas, the connected leads, and the corresponding coupling parameters.

## Numerical method

Transport properties are computed using quantum master equations implemented through the `qmeq` solver library.

The workflow is:

1. Construct the Hamiltonian and coupling structure of the Majorana box system
2. Generate the corresponding model representation for the solver
3. Solve the transport problem numerically using `qmeq`
4. Post-process the results to analyse transport behaviour across parameter regimes

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
