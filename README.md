# ML Foundations

This is a collection of practical notebooks to revisit core machine learning concepts and the mathematical foundations behind them. Each project is minimal using mostly NumPy. LLMs were used only to lightly improve written explanations - absolutely no LaTeX, code, or pseudocode was AI-generated.

## Educational Projects

### Mathematical Foundations
- [x] Matrix decomposition (orthogonal projections, eigendecomposition, SVD, PCA)
- [x] Gradient and Jacobian visualizer for multivariate functions
- [x] Backpropagation via matrix calculus (differentials, trace trick, chain rules, layer/loss derivations)
- [x] Numerical optimization (gradient descent, SGD, Momentum, Adam)

### Deep Learning
- [x] Minimal autodiff engine with multi-layer perceptron on MNIST (Tensor class, reverse-mode backprop, optimizers)
- [x] Minimal Transformer architecture implementation (attention mechanisms, multi-head attention, residuals, LayerNorm, GELU)
- [ ] Minimal Diffusion model on MNIST (forward noising + reverse denoising, denoising network, ELBO/KL derivation)
- [ ] CNN layers and training loops on toy data (BatchNorm, weight initialization, activation comparisons)

### Classical Machine Learning
- [x] Linear Regression from scratch (MLE, Bayesian inference, gradient descent, uncertainty visualization)
- [x] Logistic Regression classifier with loss surface plots and decision boundaries  
- [ ] Gradient Boosted Trees with XGBoost (boosting theory, regularization, tabular benchmarks)

## Tech Stack & Tools
- Python, NumPy, Matplotlib
- Jupyter Notebooks
- PyTorch
- Markdown + LaTeX for writeups
