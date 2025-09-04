# ML Foundations

This is a collection of practical notebooks to revisit core machine learning concepts and the mathematical foundations behind them. Each project is minimal using mostly NumPy. LLMs were used only to lightly improve written explanations - absolutely no LaTeX, code, or pseudocode was AI-generated.

## Educational Projects

### Mathematical Foundations
- [x] Matrix decomposition (SVD, eigendecomposition, orthogonal projections, PCA)
- [x] Gradient and Jacobian visualizer for multivariate functions
- [x] Backpropagation via matrix calculus (differentials, trace trick, chain rules, core layer/loss derivations)
- [ ] Numerical optimization (gradient descent, SGD, Momentum, Adam, convergence theory)

### Deep Learning
- [ ] Minigrad autodiff engine with MLP experiments (autodiff core, gradient checks, optimizers, toy datasets)
- [ ] CNN layers and training loops on toy data (BatchNorm, weight initialization, activation comparisons)
- [ ] Minimal Transformer architecture implementation (attention mechanisms, multi-head attention, residuals, LayerNorm, GELU)
- [ ] Minimal Diffusion model on MNIST (forward noising + reverse denoising, denoising network, ELBO/KL derivation)

### Classical Machine Learning
- [x] Linear Regression from scratch (MLE, Bayesian inference, gradient descent, uncertainty visualization)
- [ ] Logistic Regression classifier with loss surface plots and decision boundaries  
- [ ] Gradient Boosted Trees with XGBoost (boosting theory, regularization, tabular benchmarks)

## Paper Reading Log

### Foundational Papers
- [ ] Attention is All You Need (2017)
- [ ] Auto-Encoding Variational Bayes (2013)
- [ ] Deep Residual Learning for Image Recognition (2015)
- [ ] Diffusion Models Beat GANs on Image Synthesis (2021)
- [ ] Distilling the Knowledge in a Neural Network (2015)
- [ ] Quantizing Deep Convolutional Networks for Efficient Inference (2016)
- [ ] Word2Vec and GloVe (2013-14)
- [ ] Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer (2017)
- [ ] Low-Rank Adaptation of Large Language Models (2021)
- [ ] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)

## Tech Stack & Tools
- Python, NumPy, Matplotlib, Seaborn
- Jupyter Notebooks for derivations and code
- PyTorch, Scikit-learn (for sanity-checks only)
- Markdown + LaTeX for writeups
