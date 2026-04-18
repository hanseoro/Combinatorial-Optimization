# Neural Combinatorial Optimization for TSP

An end-to-end machine learning system for solving the **Traveling Salesman Problem (TSP)** using neural networks.

## Objectives

- Build a **neural solver** for Euclidean TSP using PyTorch
- Benchmark against classical baselines:
  - Random tour
  - Nearest neighbor
  - 2-opt local search
- Evaluate:
  - Tour length
  - Optimality gap
  - Inference latency
  - Generalization across graph sizes
- Implement **end-to-end ML pipeline**:


## Approach
### Data
- Randomly generated **Euclidean TSP instances**
- Configurable number of cities (e.g., 20, 50, 100)

### Models explored
- Attention-based neural network (Transformer / Pointer Network)

### Training
- Supervised learning using heuristic-generated tours for smaller number of nodes
- Reinforcement learning using distance as learning objective for larger number of node

### Evaluation
- Compare neural model against classical solvers
- Analyze trade-offs between:
  - solution quality
  - inference speed

