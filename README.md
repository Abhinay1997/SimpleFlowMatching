The goal of this project is to build a simple proof of concept to test out the following:

1. Different optimizers: ADAM, ADOPT, PSGD, and variants with Caution, MARS etc.
2. Different architectures: DiT, MMDiT, DiT + MMDiT blocks combined
3. Different attention implementations: SDPA, SigmoidAttention, LinearAttention as proposed by MIT-SANA
4. n-bit training tests. COAT to see if further compression works
5. Does REPA work ?
6. Kernels: qkv fusion, torch compile, matmul speedups, custom triton kernels. This is not a high priority atp.

Biases:
1. Nothing Ever Happens and Scaling compute >>> everything else
2. The more quantisable the model, the less trained it is. https://arxiv.org/pdf/2411.04330
3. DiT's tend to have grouped representation space across layers the larger they get. This is also why they are pretty robust to layer skips (for middle layers.)


Busy with Uni applications but plan on starting by the new year at least. This is also mainly for me to cross the mental barrier of the notion that training models from scratch is black magic.

Experiment Log:
##### 18/12/2024
1. dit_train.py:
   1. Flow matching on MNIST. uniform timestep, fixed lr. AdamW, fp32, class conditioned for 100 epochs, batch size 1024
   2. Issue: loss starts at 1.92 (epoch1) and oscillates around 1(epoch100.).
   3. To Try: remove class labels and train
   4. What Failed: different architectuers, my code and DiT code, excluding 1 from timesteps, different learning rates, 1e-3 to 1e-5, MNIST and FashionMNIST.

##### 20/12/2024
After a couple of hours of debugging, I found the stupid bug. In the training loop instead of calling `model(x_noisy,y,t)` I was calling `model(x,y,t)` a.k.a I was giving my model a clean image and asking it to predict the noise at that timestep. Facepalm moment.
