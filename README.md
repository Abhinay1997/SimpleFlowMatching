Finally got around to doing my FlowMatching expts:

MeanFlow + Boundary Rectification + Optimal diffusion loss formulation
1. Mnist expts
2. Replicate for cifar10

Ablations:
1. Baseline -> Standard RF 
2. Standard RF + Boundary Rectification
3. Standard RF + Optimal diffusion loss
4. Standard RF + Boundary Rectification + Optimal diffusion loss
5. MeanFlow
6. MeanFlow + Boundary Rectification
7. MeanFlow + Optimal diffusion loss
8. MeanFlow + Boundary Rectification + Optimal diffusion loss
9. Old diffusion coupling too for completeness ?

For now I'll be running them on MNIST.

Finding optimal model capacity:
-> Using batch overfit tests
