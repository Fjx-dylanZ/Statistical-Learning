---
title: "Problem Set #1"
author: "Kevin McAlister"
date: "January 12th, 2022"
output:
  prettydoc::html_pretty:
    df_print: kable
    theme: leonids
    highlight: github
    toc: no
    toc_depth: 2
    toc_float:
      collapsed: no
  pdf_document:
    toc: no
    toc_depth: '2'
urlcolor: blue
---

```{r, include=FALSE}
library(ggplot2)
library(data.table)
knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning = FALSE, fig.width = 16/2, fig.height = 9/2, tidy.opts=list(width.cutoff=60), tidy=TRUE)
```

This is the first problem set for QTM 385 - Intro to Statistical Learning.  It includes both analytical derivations and computational exercises.  While you may find these tasks challenging, I expect that a student with the appropriate prerequisite experience will be able to complete them

Please use the intro to RMarkdown posted in the Intro module and my .Rmd file as a guide for writing up your answers.  You can use any language you want, but I think that a number of the computational problems are easier in R.  Please post any questions about the content of this problem set or RMarkdown questions to the corresponding discussion board.

Your final deliverable should be a .zip archive that includes a .Rmd/.ipynb file and either a rendered HTML file or a PDF.  Students should complete this assignment **on their own**.  This assignment is worth 10% of your final grade.

This assignment is due by January 26th, 2022 at 6:00 PM EST.  


# Problem 1 (40 pts.)

Linear regression is a fundamental tool for statistics and machine learning.  At its core, linear regression is a simple task: given a set of $P$ predictors, $\{\mathbf{x}_i\}_{i = 1}^N = \boldsymbol{X}$, with each $\mathbf{x}_i$ a $P + 1$-vector of predictors with a 1 as the first element (to account for an intercept) and outcomes, $\{y_i\}_{i = 1}^N = \boldsymbol{y}$, find the $P + 1$-vector $\hat{\boldsymbol{\beta}}$ that minimizes the residual sum of squares:

$$\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}^*}{\text{argmin}} \left[ \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right)' \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right) \right]$$

This can also be expressed as a summation over errors:

$$\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}^*}{\text{argmin}} \left[ \sum \limits_{i = 1}^N \left(y_i - \boldsymbol{\beta}^{*'} \boldsymbol{x}_i \right)^2 \right]$$

In the case of a single predictor, **simple linear regression** can be easily expressed without matrix notation:

$$\{\hat{\alpha} , \hat{\beta}\} = \underset{\alpha^*, \beta^*}{\text{argmin}} \left[ \sum \limits_{i = 1}^N \left(y_i - \alpha^* - \beta^* x_i  \right)^2 \right]$$

## Part 1 (8 pts.)

Linear regression was widely used when computers were dinky little calculators with only 512 MBs of RAM (I remember upgrading my first PC to a 1 GB stick of RAM and wondered how anyone would ever need more!) because it admits a slew of **analytical solutions** to the above minimization problems.  

Let's start with simple linear regression.  Find the values of $\hat{\alpha}$ and $\hat{\beta}$ that **minimize** the residual sum of squares.  Try to reduce these as much as possible (using the identities for covariance and variance *hint, hint*).

Some hints:

  1. I think it's easier to start with $\alpha^*$.
  2. Remember that the empirical covariance estimator is $\frac{\sum \limits_{i = 1}^N (x_i - \bar{X})(y_i - \bar{Y})}{N}$ and that the empirical variance estimator is $\frac{\sum \limits_{i = 1}^N (x_i - \bar{X})^2}{N}$.  You can do the same thing with $N - 1$ (to ensure that the estimator is always unbiased, not just asymptotically).
  3. Remember that the regression coefficient is defined as $\frac{\text{Cov}(X,Y)}{\text{Var}(X)}$.  You should probably get the same thing here for one of the parts.
  4. To make everything easier, try to get sums to averages (divide by $N$ when needed)

### Part 1 Solution

Consider the squared error for SLR,
$$
\begin{align*}
\sum \limits_{i = 1}^N \left(y_i - \alpha^* - \beta^* x_i  \right)^2
\end{align*}
$$
To find the $a^*$ and $b^*$ that minimizes the above equation, we first derive its first-order condition.
$$
\begin{align*}
\frac{\partial}{\partial a^*} \sum \limits_{i = 1}^N \left(y_i - \alpha^* - \beta^* x_i  \right)^2 &= -2\sum \limits_{i = 1}^N \left(y_i - \alpha^* - \beta^* x_i  \right)&=0\\
&=\sum\limits_{i=1}^N y_i - \sum\limits_{i=1}^N a^* - \beta^*\sum\limits_{i=1}^N x_i &=0\\ \\
\Longrightarrow \quad \quad \sum\limits_{i=1}^N a^* &=\sum\limits_{i=1}^N y_i - \beta^*\sum\limits_{i=1}^N x_i\\
\frac{1}{N}\sum\limits_{i=1}^N a^* &=\frac{1}{N}\sum\limits_{i=1}^N y_i - \frac{1}{N}\beta^*\sum\limits_{i=1}^N x_i\\
a^* &= \bar y-\beta^* \bar x
\end{align*}
$$
$$
\begin{align*}
\frac{\partial}{\partial \beta^*} \sum \limits_{i = 1}^N \left(y_i - \alpha^* - \beta^* x_i  \right)^2 &= -2\sum \limits_{i = 1}^N x_i\left(y_i - \alpha^* - \beta^* x_i  \right)&=0\\
&= \sum\limits_{i=1}^Nx_i(y_i-\alpha^*-\beta^*x_i) &=0\\
&= \sum\limits_{i=1}^N x_i(y_i-\bar y - \beta^*(x_i-\bar x) &=0\\
\Longrightarrow\quad \quad &\sum\limits_{i=1}^N x_i(y_i - \bar y) - \beta^*\sum\limits_{i=1}^Nx_i(x_i - \bar x) &=0\\
\beta^* &=\frac{\sum\limits x_i(y_i-\bar y)}{\sum\limits x_i(x_i-\bar x)}\\
&=\frac{\sum\limits x_iy_i -\sum\limits x_i \bar y}{\sum\limits x_i^2 -\sum\limits x_i \bar x}\\
&=\frac{\sum\limits x_iy_i-N\bar x\bar y}{\sum\limits x_i^2 - N\bar x^2} \tag{1}
\end{align*}
$$
Then, consider the following equivalent equations:
$$
\begin{align*}
\sum_{i=1}^N (x_i - \bar x)(y_i - \bar y) &= \sum\limits_{i=1}^N(x_iy_i- x_i \bar y -\bar x y_i + \bar x \bar y)\\
&=\sum\limits_{i=1}^N x_iy_i -N \bar x\bar y - N\bar x \bar y + N\bar x \bar y\\
&=\sum\limits_{i=1}^N x_i y_i - N \bar x \bar y
\end{align*}
$$
$$
\begin{align*}
\sum\limits_{i=1}^N (x_i-\bar x)^2 &= \sum\limits_{i=1}^N(x_i^2-2x\bar x+ \bar x ^2)\\
&= \sum\limits_{i=1}^N x_i^2 -2N\bar x^2 + \bar Nx^2\\
&= \sum\limits_{i=1}^N x_i^2 - N\bar x^2
\end{align*}
$$
Therefore, we can rewrite (1) to
$$
\begin{align*}
	\beta^*&=\frac{\sum\limits_{i=1}^N (x_i-\bar x)(y_i - \bar y)}{\sum\limits_{i=1}^N(x_i-\bar x)^2}\\
	&= \frac{N\text{Cov}(X, Y)}{N\text{Var}(X)}\\
&=\frac{\text{Cov}(X,Y)}{\text{Var}(X)}
\end{align*}
$$

## Part 2 (7 pts.)

A common theme we'll see this semester is the notion of optimization - finding the values of parameters that maximize/minimize objective functions of interest.  Optimization can be tricky when functions are not strictly convex/concave - most computational methods of optimization can only guarantee that they locate one of potentially many **local optima** when we really want to find the **global optimum**.  Argue that the sum of squared errors function for simple linear regression is **strictly convex** in $\alpha^*$ and $\beta^*$ and that there exists a unique and finite global minimum.  You can assume that there is nothing funky here (e.g. variance in both X and Y, $N \ge 2$, etc.).



### Part 2 Solution

Consider the matrix form of simple linear regression,
$$
\begin{align*}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}^*}{\text{argmin}} \left[ \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right)' \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right) \right]
\end{align*}
$$
where $\beta^* = \begin{bmatrix} \alpha^* \\ \beta^* \end{bmatrix}$, $\boldsymbol{y} \in \mathbb{R}^n$, $\boldsymbol{X}=\begin{bmatrix}1 & x_1 \\ \vdots & \vdots \\ 1 & x_N \end{bmatrix}\in \mathbb{R}^{N\times 2}$
Let $f(\boldsymbol{X}, \boldsymbol{\beta})= (\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)^\top(\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)$ .  From *part 4*, we have derived the following normal equation:
$$
\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{\beta}^* = \boldsymbol{X}^\top \boldsymbol{y}
$$
Because $\boldsymbol{X}^\top \boldsymbol{X}$ is symmetric, $\boldsymbol{X}^\top \boldsymbol{X} \succeq 0$. In this case, $\boldsymbol{X}$ is full-rank, so $\boldsymbol{X}^\top \boldsymbol{X} \succ 0$
If $\boldsymbol{X}^\top \boldsymbol{X} \succ 0$, then it has positive eigenvalues, which makes it invertible. Therefore the following system has a **unique** solution.
$$
(\boldsymbol{X}^\top \boldsymbol{X})\boldsymbol{\beta}^* = \boldsymbol{X}^\top \boldsymbol{y}
$$
## Part 3 (10 pts.)

Coding and statistical learning go hand-in-hand.  Your previous classes have largely been pen-and-paper focused - even your regression class likely only covered methods that could, in theory, be done with some pen, paper, and a calculator (though inverting a large matrix by hand would be considered cruel and unusual torture by most).  We're going to quickly move out of the realm of methods that are analytically solvable, so understanding how **algorithms** work from the ground up will help you to understand why something works when we can't always prove it via mathematics.  In many cases, too, we'll find that computational approaches with no pen-and-paper analogue (cross-validation, bootstrapping, black-box optimization, etc.) will provide superior answers to the analytical methods.^[Have you ever really thought about how a computer inverts a matrix?  It requires a lot of mathematics that aren't needed when thinking about doing it by hand.  There's tricks and decompositions that make it work almost instantaneously!  This is just one example of a situation where the computational approach is far superior to the analytical one]

This said, let's assume that you didn't know how to find the optimum derived in part 1.  We could always use **numerical optimization** methods to find the minimum.  Additionally, since we can leverage our knowledge that the function is strictly convex in $\alpha$ and $\beta$, we should always land on the same answer!  So, this method will be equivalent.

Write a function called `sse_minimizer` that uses a built-in optimization routine to find the values of $\alpha$ and $\beta$ that minimize the sum of squared errors for simple linear regression.  `sse_minimizer` should take in two arguments: `y` - a $N$-vector of outcome values and `x` - a $N$-vector of predictor values.  It should return a list (or equivalent holder) with five elements elements: 1) `alpha_est` - the value of the intercept given by the optimization routine, 2) `beta_est` - the value of the regression coefficient given by the optimization routine, 3) `alpha_true` - the true value of the intercept computed using your answer from part 1, 4) `beta_true` - the value of the regression coefficient computed using your answer from part 1, and 5) `mse` - the mean squared error (the sum of squared errors divided by the number of observations) between the predicted value of the outcome and the true value of the outcome.

To test your function, generate some simulated data:

  1. Generate 1000 uniform random values between -3 and 3 as `x`
  2. Choose some values for the intercept and slope, `a` and `b`.  Using `x`, generate `y` as `a + b*x`.
  3. Add some random noise to `y` - `y <- y + rnorm(100,0,1)` for example.
  4. Plug `x` and `y` into your function and see if it returns the correct parameter values.
  
A full-credit answer will demonstrate that the function works!

Some tips to help you get going (in R, if that's your choice; Python is similar):

  1. Write a function called `sum_squared_errors` that takes three arguments: 1) `coefficients` - a 2-vector of coefficients ($\alpha$ and $\beta$, respectively, 2) `y` - a $N$-vector of outcome values, and 3) `x` - a $N$-vector of predictor values.  The function should return the sum of squared errors **given** the input values of $\alpha$ and $\beta$.
  
  2. Call `sum_squared_errors` from `sse_minimizer` within the optimization routine.  I recommend using base R's `optim` to find the minimum.  `optim` can be a bit confusing on the first go, so be sure to **read the documentation** and look for examples online if you're confused.  I'm also happy to help in office hours, but I think this is an important "do-it-yourself" technique.
  
  3. Lists are nifty because we can easily name elements.  If we have `lst <- list()`, then we can assign `lst$alpha <- foo` and `lst$beta <- bar`.



### Part 3 Solution

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math

# Generate data from uniform distribution

np.random.seed(777)
x_train = np.random.uniform(-3, 3, 1000)
alpha_choice = math.pi
beta_choice = math.e
y_train = alpha_choice + beta_choice * x_train + np.random.normal(0, 1, 1000)

def sum_squared_errors(params):

"""
Function to calculate the sum of squared errors for a simple linear regression model.
Inputs:
params: 1D array of weights
Outputs:
sse: scalar value of sum of squared errors
"""

# Get the weights
w = params
# Get the X matrix
X = np.array([np.ones(len(x_train)), x_train]).T
# Calculate the sum of squared errors
sse = np.sum(np.square(X.dot(w) - y_train))
return sse

  

def sse_minimizer(y, x):
"""
Function to find the optimal coefficients for a simple linear regression model.
Inputs:
y: 1D array of training data
x: 1D array of training data
Outputs:
w: 1D array of weights
"""
# Optimization method with scipy.optimize.minimize
# initialize w
w = np.array([0, 0])
opt_result = optimize.minimize(sum_squared_errors, w)
alpha_est, beta_est = opt_result.x
beta_true = (sum(x*y)-len(x)*np.mean(x)*np.mean(y))/(sum(x**2)-len(x)*np.mean(x)**2) # from part 1
alpha_true = np.mean(y)-beta_true*np.mean(x) # from part 1
return {"alpha_est": alpha_est, "beta_est": beta_est, "alpha_true": alpha_true, "beta_true": beta_true}

sse_minimizer(y_train, x_train)



```

```output
{'alpha_est': 3.150021400640548,  'beta_est': 2.728723321003393,  'alpha_true': 3.1500214119019394,  'beta_true': 2.728723328567721}
```

## Part 4 (7 pts.)

Now, let's move on to multiple linear regression.  Using the same logic as above, we can show that the sum of squared errors is strictly convex in $\hat{\boldsymbol{\beta}}$, so there exists a unique minimum (assuming $\boldsymbol{X}$ is of full rank).  Working with matrix derivatives, show that the $\hat{\boldsymbol{\beta}}$ that minimizes the sum of squared errors is:

$$\boldsymbol{\hat \beta}=\underset{\boldsymbol{\beta}^*}{\text{argmin}} \left[ \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right)' \left(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}^*  \right) \right] = \left(\boldsymbol{X}'\boldsymbol{X} \right)^{-1} \boldsymbol{X}'\boldsymbol{y}$$

### Part 4 Solution

Let $f(\boldsymbol{X}, \boldsymbol{\beta})= (\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)^\top(\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)$ and we can derive its gradient as the following
$$
\begin{align*}
f(\boldsymbol{X}, \boldsymbol{\beta}) &=(\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)^\top(\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\beta}^*)\\
&=(\boldsymbol{y}^\top-\boldsymbol{\beta}^\top \boldsymbol{X}^\top)(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}^*)\\
&= \boldsymbol{y}^\top \boldsymbol{y} - \boldsymbol{y}^\top \boldsymbol{X} \boldsymbol{\beta}^* - {\boldsymbol{\beta}^*}^\top \boldsymbol{x}^\top \boldsymbol{y} + {\boldsymbol{\beta}^*}^\top (\boldsymbol{X}^\top \boldsymbol{X}) \boldsymbol{\beta}^*\\
&=\boldsymbol{y}^\top \boldsymbol{y}-({\boldsymbol{\beta}^*}^\top \boldsymbol{X}^\top \boldsymbol{y})^\top - {\boldsymbol{\beta}^*}^\top \boldsymbol{X}^\top \boldsymbol{y} + {\boldsymbol{\beta}^*}^\top (\boldsymbol{X}^\top \boldsymbol{X}) \boldsymbol{\beta}^*\\
&=\boldsymbol{y}^\top \boldsymbol{y} - 2{\boldsymbol{\beta}^*}^\top \boldsymbol{X}^\top \boldsymbol{y}+ {\boldsymbol{\beta}^*}^\top (\boldsymbol{X}^\top \boldsymbol{X}) \boldsymbol{\beta}^* & \text{because ${\boldsymbol{\beta}^*}^\top \boldsymbol{X}^\top \boldsymbol{y} \in \mathbb{R}$}
\end{align*}
$$
$$
\begin{align*}
\nabla_\boldsymbol{\beta} f(\boldsymbol{X}, \boldsymbol{\beta}^*)&= -2 \boldsymbol{X}^\top \boldsymbol{y} + 2\boldsymbol{X}^\top \boldsymbol{X}\boldsymbol{\beta}^*\\
&=2(\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{\beta}^* - \boldsymbol{X}^\top \boldsymbol{y})\\
&=\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{\beta}^*-\boldsymbol{X}^\top \boldsymbol{y} &=0\\
\end{align*}
$$
$$
\begin{align*}
\Longrightarrow \quad \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{\beta}^* &= \boldsymbol{X}^\top \boldsymbol{y}\\
\boldsymbol{\beta}^* &= (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{y}
\end{align*}
$$

## Part 5 (8 pts.)

Frequently, we seek to perform **inference** on the values of $\boldsymbol{\beta}$ - we want to determine if the noise associated with the OLS estimator is small enough to say that each $\beta_j$ is statistically different from zero.  First, we want to show that the OLS estimator is **unbiased** for the true value of $\boldsymbol{\beta}$ so that we can claim that our inference is meaningful.  Then, we need to derive the **standard error** of the estimator to perform inference.    

Show that $\hat{\boldsymbol{\beta}} = \left(\boldsymbol{X}'\boldsymbol{X} \right)^{-1} \boldsymbol{X}'\boldsymbol{y}$ is unbiased for $\boldsymbol{\beta}$,  $E[\boldsymbol{\hat{\beta}}] = \boldsymbol{\beta}$.  Then, derive the **variance-covariance matrix** for $\boldsymbol{\hat{\beta}}$ - $\text{Cov}[\hat{\boldsymbol{\beta}}]$.  The square root of the diagonal of the variance-covariance matrix then provides the **standard errors**.

Some helpful identities:

  1. For multiple linear regression, we assume that $\boldsymbol{X}$ is constant while $E[\boldsymbol{y}] = X\boldsymbol{\beta}$ and $\text{Cov}[\boldsymbol{y}] = \sigma^2 \mathcal{I}_{N}$ where $\sigma^2$ is the constant error variance (e.g. a scalar) and $\mathcal{I}_N$ is the $N \times N$ identity matrix.
  2. Suppose we want to know $\text{Cov}[\boldsymbol{A}\boldsymbol{y}]$ where $\boldsymbol{A}$ is a matrix of constants and $\boldsymbol{y}$ is a random vector.  Then $\text{Cov}[\boldsymbol{A}\boldsymbol{y}] = \boldsymbol{A} \times \text{Cov}[\boldsymbol{y}] \times \boldsymbol{A}'$   
  
### Part 5 Solution
By definition,  $\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}+\boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon}$ denotes residual.
$$
\begin{align*}
\hat{\boldsymbol{\beta}} &= \left(\boldsymbol{X}^\top\boldsymbol{X} \right)^{-1} \boldsymbol{X}^\top\boldsymbol{y}\\
&= \left(\boldsymbol{X}^\top\boldsymbol{X} \right)^{-1} \boldsymbol{X}^\top(\boldsymbol{X}\boldsymbol{\beta}+\boldsymbol{\epsilon})\\
&= (\boldsymbol{X^\top X})^{-1} \boldsymbol{X^\top}\boldsymbol{X\beta}+ (\boldsymbol{X ^\top X})^{-1}\boldsymbol{X}^\top \boldsymbol{\epsilon}\\
&= \mathcal{I}_K \boldsymbol{\beta} + (\boldsymbol{X ^\top X})^{-1}\boldsymbol{X ^\top \epsilon} && \text{K coefficients}\\
E[\boldsymbol{\hat \beta}]=E[\boldsymbol{\hat \beta}|X]&= E(\boldsymbol{\beta}) + E[(\boldsymbol{X ^\top X})^{-1}\boldsymbol{X ^\top \epsilon}|X]\\
&=\boldsymbol{\beta}+ (\boldsymbol{X ^\top X})^{-1}\boldsymbol{X ^\top} E[\boldsymbol{\epsilon}|X]&& E(\epsilon_i|x_i)=0\\ 
E[\boldsymbol{\hat \beta}]&= \boldsymbol{\beta} 
\end{align*}
$$
$$
\begin{align*}
\text{Cov}[\boldsymbol{y}]=\sigma^2\mathcal{I}_N
\end{align*}
$$
By definition, $\text{Cov}(\boldsymbol{\hat \beta})=E[(\boldsymbol{\beta} - E[\boldsymbol{\beta}])(\boldsymbol{\hat \beta}-E[\boldsymbol{\beta}]) ^\top]$
$$
\begin{align*}
\text{Cov}(\boldsymbol{\hat \beta})&= E\left[(\boldsymbol{\hat \beta} - E[\boldsymbol{\beta}])(\boldsymbol{\hat \beta}-E[\boldsymbol{\beta}]) ^\top \right]\\
&=E\left[(\boldsymbol{\hat \beta}- \boldsymbol{\beta})(\boldsymbol{\hat \beta}- \boldsymbol{\beta})^\top\right]\\
&=E\left[\left(\left(\boldsymbol{X}^\top\boldsymbol{X} \right)^{-1} \boldsymbol{X}^\top\boldsymbol{y} - \boldsymbol{\beta}\right)\left(\left(\boldsymbol{X}^\top\boldsymbol{X} \right)^{-1} \boldsymbol{X}^\top\boldsymbol{y} - \boldsymbol{\beta}\right)^\top\right]\\
&=E\left[\left(\left(\boldsymbol{X^\top X}\right)^{-1}\boldsymbol{X}^\top\left(\boldsymbol{X \beta+\epsilon}\right)-\boldsymbol{\beta}\right)\left(\left(\boldsymbol{X^\top X}\right)^{-1}\boldsymbol{X}^\top\left(\boldsymbol{X \beta+\epsilon}\right)-\boldsymbol{\beta}\right)^\top\right]\\
&\ \ \vdots && \text{part 5 (1)}\\ 
&= E\left[\left(\boldsymbol{\beta}+\left(\boldsymbol{X ^\top X}\right)^{-1}\boldsymbol{X}^\top \boldsymbol{\epsilon}-\boldsymbol{\beta}\right)\left(\boldsymbol{\beta}+\left(\boldsymbol{X ^\top X}\right)^{-1}\boldsymbol{X}^\top \boldsymbol{\epsilon}-\boldsymbol{\beta}\right)^\top\right]\\
&=E\left[\left(\boldsymbol{X ^\top X}\right)^{-1}\boldsymbol{X ^\top}\boldsymbol{\epsilon \epsilon ^\top}\boldsymbol{X}\left(\boldsymbol{X ^\top X}\right)^{-1}\right]\\ 
&= \dots?
\end{align*}
$$
$\text{I am not sure how to do this part}$


# Problem 2 (40 pts.)

Over the semester, we're going to leverage probability **distributions** and common summaries of probability distributions - expectations, variance, covariance, etc.  The goal of this problem is to review what you've already learned in previous classes and (perhaps) introduce the idea of **simulation** to understand properties of distributions.

Suppose that we have a random variable $x$ such that:
$$f(x) \sim \exp[-\lambda x] \text{ if } x \ge 0 \text{ else } f(x) = 0$$
where $\sim$ means **distributed as** and $\lambda$ is an arbitrary parameter value greater than 0.  Furthermore, we know that $x$ can only take values on the positive real line so the **density** of any $x$ less than zero is exactly 0.

## Part 1 (7 pts.)

As is, the probability distribution above is not a proper probability density function - it doesn't integrate to 1!  Given the above info, show that the value of $Z$ that **normalizes** the density function to a proper probability density function:
$$f(x) = Z \times \exp[-\lambda x] \text{ if } x \ge 0 \text{ else } f(x) = 0$$
is $\lambda$.

### Part 1 Solution

Because $f(x) = 0 \ \forall x<0$, we integrate it over the interval $[0, \infty)$ 
$$
\begin{align*}
\int_0^\infty \lambda\exp[-\lambda x]dx &= -\exp(-\lambda x)\Big|^\infty_0\\
&= \lim_{x\rightarrow\infty} [-\exp(\lambda x)]+\exp(0)\\
&= 0+1\\
&=1
\end{align*}
$$


## Part 2 (8 pts.)

Given the proper PDF above, derive the **expected value** and **variance** of $x$.


### Part 2 Solution

$$
\begin{align*}
E[X] &= \int_{-\infty}^\infty xf(x)dx\\
&= \int_{0}^\infty x\lambda\exp(-\lambda x)dx \\
&= \frac{1}{\lambda}\\
\text{Var}[X] &=\int_{-\infty}^\infty [x-E(X)]^2f(x)dx\\
&= \int_{0}^\infty [x-\frac{1}{\lambda}]^2\lambda\exp(-\lambda x)dx\\
&= \frac{1}{\lambda^2}
\end{align*}
$$

## Part 3 (7 pts.)

Show that the corresponding cumulative density function for the above PDF is $1 - \exp[-\lambda x]$ and that the median of this distribution is $\frac{\text{ln}(2)}{\lambda}$.

### Part 3 Solution

$$
\begin{align*}
\text{CDF:} \quad \int_0^x \lambda \exp(-\lambda t)dt&=\lambda\int^x_0 \exp(-\lambda t)dt\\
&= \lambda\left(-\frac{1}{\lambda}\left[e^u\right]\Big|^{-\lambda x}_0\right) &\text{u-sub: $u=-\lambda x$}\\
&= -(\exp(-\lambda x) - 1)\\
&= 1-\exp(-\lambda x)
\end{align*}
$$
$$
\begin{align*}
\text{median:}\quad 1-\exp(-\lambda x)&=0.5\\
\exp(-\lambda x)&=0.5\\
\ln(\exp(-\lambda x))&=\ln(0.5)\\
-\lambda x&=\ln(1/2)\\
-\lambda x&=\ln(1)-\ln(2)\\
x&= \frac{\ln{2}}{\lambda}
\end{align*}
$$

## Part 4 (8 pts.)

The most common scenario where we'll see probability distributions is when trying to estimate the parameters that dictate a data generating process.  For example, we may observe $N$ observations that are assumed to independent and identically distributed draws from the above probability distribution.  Since we've observed these draws and made the i.i.d. assumption, we can derive a **likelihood** function for the data:
$$f(X ; \lambda) = \prod \limits_{i = 1}^N \lambda \exp[-\lambda x_i]$$
where $x_i$ is one of the $N$ observations ($x_i \in \{x_1,x_2,...,x_{N-1},x_{N}\}$).  Find the value of $\lambda$ that maximizes the above likelihood function (e.g. the maximum likelihood estimator of $\lambda$) given $N$ observations.

Some hints:

  1. Start by simplifying the product as much as possible.  Remember that a constant, $a$, times itself $M$ times is $a^M$.  Also, don't forget the rules of exponents!  Finally, anything not subscripted is a constant - use that to your advantage! 
  2. Take a natural log of the simplified likelihood function - this is called the **log-likelihood**.  Remember that the log of a product is equal to the sum of the logs.  This is a good step because it gets rid of the annoying exponentials.  We can also do this because a log is a **one-to-one** transformation, so it preserves the maximum.
  3. Use calculus to find the value of $\lambda$ that maximizes the expression.  If this solution is unique, use a second derivative test to show that you have found a maximum or logically argue that the original or log likelihood is strictly concave in $\lambda$.

### Part 4 Solution

$$
\begin{align*}
f(X ; \lambda) &= \prod \limits_{i = 1}^N \lambda \exp[-\lambda x_i]\\
&=\lambda^N \exp\left[-\lambda\left(\sum\limits_{i=1}^N x_i\right)\right]\\
\Longrightarrow\quad\ln(f(X;\lambda))&=\ln(\lambda^N )+\left(-\lambda \sum\limits_{i=1}^N x_i\right)\\
&=N\ln(\lambda)-\lambda\sum\limits_{i=1}^N x_i

\end{align*}
$$
$$
\begin{align*}
\frac{\partial \ln f(X;\lambda)}{\partial \lambda} &= \frac{N}{\lambda}-\sum\limits_{i=1}^N x_i &= 0\\
\Longrightarrow \quad \quad  \lambda^* &= \frac{N}{\sum\limits_{i=1}^Nx_i } & \text{MLE}\\
\frac{\partial^2 \ln f(X;\lambda)}{\partial \lambda^2} &= -\frac{N}{\lambda^2}\\
\frac{\partial^2 \ln f(X;\lambda^*)}{\partial \lambda^2} &= \frac{-N}{\left(\frac{N}{\sum\limits_{i=1}^Nx_i }\right)^2}= - \frac{\left(\sum\limits_{i=1}^Nx_i\right)^2}{N} &< 0 &&\because x_i\geq0 \ \forall i
\end{align*}
$$
Strictly convex in $\lambda$ because there is only one solution for the first order condition, and the solution is a maximum, therefore the global maximum.
## Part 5 (10 pts.)

Maximum likelihood estimators are an important part of statistics and machine learning as they are commonly used to minimize certain types of **loss functions** - find the parameter that maximizes the likelihood/minimizes the loss with respect to the data generating process.  MLEs are desirable in statistics because they have a number of desirable properties like asymptotic unbiasedness, consistency, and efficiency and follow the same generic recipe regardless of data type.  Asymptotic unbiasedness means that the estimator converges to the correct answer almost surely as $N \rightarrow \infty$. Asymptotic consistency implies that as $N \rightarrow \infty$, the standard deviation of the sampling distribution (e.g. the standard error) goes to zero.  Asymptotic efficiency implies that the estimator achieves the lowest possible variance on the path to zero - otherwise known as the Cramer-Rao lower bound.

For this last part, I want you to graphically show the asymptotic consistency and efficiency properties using a method called **parametric bootstrapping**.  We're going to discuss bootstrapping as a method of uncertainty calculation and this part is a short introduction to the method.  Bootstrapping techniques leverage the fact that probability can be interpreted as the long run proportion of occurrence.  We can replicate long-run frequencies by using computational methods to take random draws from a distribution.

The distribution you've been working with here is called the **exponential distribution** and is a key distribution in the study of random counting processes and Bayesian statistics.  This makes the process of taking **random draws** from the distribution easy - just use `rexp()` in R and the equivalent functions in other languages!  The main gist of bootstrapping is that we can replicate the process of repeated sampling by taking a large number of random draws from the distribution of interest to estimate the quantity of interest.

Write a function called `bootstrap_se` that takes in three arguments: 1) `n` - the sample size, 2) `b` - the number of bootstrap replicates, and 3) `lambda` - a value for $\lambda$.  Your function should do the following:
```{r, eval=FALSE}
For n, b, and lambda
  For i in 1:b
    Draw n values from Exp(lambda)
    Compute MLE for lambda
    Store MLE
  Compute standard deviation of MLEs
  Return standard deviation
```
In words, this function should take $N$ samples from an exponential distribution parameterized by $\lambda$ and compute the MLE implied by your random draws $B$ times.  The standard deviation of these $B$ values converges almost surely to the standard error of the sampling distribution for the MLE which is used to perform inference about the value of the parameter.

Using your function, evaluate the standard deviation of the MLE setting `n` equal to 10,20,30,....,280,290,300, `b` equal to 250, and `lambda` equal to 1/3.  Then, plot the standard deviations against the values of `n` as a line graph.  Label this as the "Bootstrap" line.  Does this line approach 0?  What is approximate rate at which it converges (think in terms of the sample size)?  

To demonstrate that the MLE is maximally efficient (the most efficient estimator possible), compute the Cramer-Rao lower bound for the exponential MLE given a value of $N$.  The Cramer-Rao lower bound, the minimum variance that can be achieved with an asymptotically unbiased estimator of $\lambda$, is shown below:
$$\text{CRLB}[f(X ; \lambda)] = \sqrt{-\left[\frac{\partial^2 \log f(X ; \lambda)}{\partial \lambda \partial \lambda}\right]^{-1}} = \frac{\lambda}{\sqrt{N}}$$
Plot this value against the corresponding values of `n` on the same plot as your bootstrapped line and label this as the "CRLB" line.  Does the bootstrapped MLE standard error reach the Cramer-Rao lower bound?  Generate the same figure for at least 2 other values of $\lambda$.  Does the same relationship hold?  

### Part 5 Solution

```python
np.random.seed(1234)
def bootstrap_se(n, b, lambdaValue):
    """
    Function to calculate the standard error of the bootstrap estimate of MLE of exponential distribution.
    Inputs:
        n: scalar value of sample size
        b: scalar value of number of bootstrap samples
        lambdaValue: scalar value of lambda
    Outputs:    
        se: scalar value of standard error
    """
    MLE_list = []
    for i in range(1, b):
        # Generate bootstrap sample
        bootstrap_sample = np.random.exponential(lambdaValue, n)
        # Calculate the MLE of exponential distribution
        mle_lambda = n / sum(bootstrap_sample) # from part 4
        MLE_list.append(mle_lambda)
    # Calculate the standard error
    se = np.std(MLE_list)
    #print("mean:{}, lambda:{}, se:{}, n:{}".format(np.mean(MLE_list), lambdaValue, se, n))
    return se

# dataframe to store the bootstrap standard errors
bootstrap_se_df = pd.DataFrame(columns=["n", "b", "lambdaValue", "se", "CRLB"])
bootstrap_se_df["n"] = np.linspace(10, 300, 30)
bootstrap_se_df["n"] = bootstrap_se_df["n"].astype(int)
bootstrap_se_df["b"] = 250
bootstrap_se_df["b"] = bootstrap_se_df["b"].astype(int)
bootstrap_se_df["lambdaValue"] = bootstrap_se_df.apply(lambda row:[1/3, 1/2, 1, 2, 3, 4], axis=1)
bootstrap_se_df = bootstrap_se_df.explode("lambdaValue")
bootstrap_se_df["se"] = bootstrap_se_df.apply(lambda row: bootstrap_se(row["n"], row["b"], row["lambdaValue"]), axis=1)
bootstrap_se_df["CRLB"] = bootstrap_se_df.apply(lambda row: row["lambdaValue"]/np.sqrt(row["n"]), axis=1)


```

```python
bootstrap_se_df
```
![df.png](https://cleanshot-cloud-fra.s3.eu-central-1.amazonaws.com/media/27554/571bW8M34M538ydHKZXjtsSR7yMrSHYfOd7IfSTi.jpeg?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIQDNApENQSvBFK34GNwLU%2Bxj9KaoSzY1X3VB%2Bz6F8V6BBgIgJSCn93HYW%2Fpwx7AySvIsq6BbTe%2FR2VQSFolza1PzG7YqqgIIvv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5MTk1MTQ0OTE2NzQiDOI3OTKknR%2FeDVpTfSr%2BAceLDOssKiwPxZuf2khqMyC2fBTiL7v72z3%2BLvdBgXzQjLTkfxKhIy69wQAAr%2F9t9Hfib0xWsan93Jrl5T3pxTg%2FcNsdOPEoDgXm0zlA%2F3jiEsZEZcm%2FiauT1BdLwTP7Wur9cnwR71JjnpBiLUmT0FjFBqfgsD4Rzq4SL2c62kaulgS9uSj8zDnhQx3gh05JgJa7qrePJyUjmMq%2B2WigEhk%2FrXQ8XgHD6dAwkVFa3Ix0GZ0o%2BIRBYgOhYg0OkPhWTIbwbqXDe2%2Fj4mrRL4OcvJSYFmco1oT9HJRvpg2Gl5zljpx8YQd2%2BPllA85%2BWWN61NcVoAseRbKBD6UDzS7KMPa2yo8GOpoB1l4p0AiBFNo2hXmX13NdX1Ve9ypfSLVA25jzdRClzEX8QrAcsfKJJDf2uHNAqlvuqNdCqjnhz7xIyp16DdPP3kDsp3Q63Ci2v1hHBujGxqM3diWgVZ%2BvevSrnKSRT0vgi5Bgn6i7MrSOG1mkZ882gvM%2FYWXI%2B0TWs%2B6ai4q%2FHNYFKE3sCmcSrclKWrTEkXVYysTCmlLd%2By3%2BeQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA5MF2VVMNBVJSD6E4%2F20220127%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220127T135447Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Signature=8a75c8684d9790314c6acb060f3c87d29dce03f9db7c9c3bcd4b1cffd6a64c2b)

```python
def plot_bootstrap_se(bootstrap_se_df, lambdaValue, subplot_row, subplot_col):
    #subset df
    bootstrap_se_df_sub = bootstrap_se_df[bootstrap_se_df["lambdaValue"] == lambdaValue]
    
    se = bootstrap_se_df_sub["se"]
    crlb = bootstrap_se_df_sub["CRLB"]
    ax[subplot_row, subplot_col].plot(bootstrap_se_df_sub["n"], se, label="Bootstrap SE")
    ax[subplot_row, subplot_col].plot(bootstrap_se_df_sub["n"], crlb, label="CRLB", linestyle="--")
    ax[subplot_row, subplot_col].set_title(f"lambda = {lambdaValue}")
    ax[subplot_row, subplot_col].set_xlabel("n")
    ax[subplot_row, subplot_col].set_ylabel("SE")
    ax[subplot_row, subplot_col].legend()

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plot_bootstrap_se(bootstrap_se_df, 1/3, 0, 0)
plot_bootstrap_se(bootstrap_se_df, 1/2, 0, 1)
plot_bootstrap_se(bootstrap_se_df, 1, 0, 2)
plot_bootstrap_se(bootstrap_se_df, 2, 1, 0)
plot_bootstrap_se(bootstrap_se_df, 3, 1, 1)
plot_bootstrap_se(bootstrap_se_df, 4, 1, 2)
plt.tight_layout()
plt.show()

```
![graph](https://cleanshot-cloud-fra.s3.eu-central-1.amazonaws.com/media/27554/j0N7VantExHzSaGx9HSlE6eTIGt7pOn8Mzi9QAaJ.jpeg?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIQDwu4pkB4FuFWAvb3A8P2iku4JVoYBNP9zZtfq4j62TTwIgYefFA4DeiHCUQU%2BuJI6zsv%2FqFUUnma5wZKpvqbZTKR8qqgIIv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5MTk1MTQ0OTE2NzQiDHbUg6WFuQSC9Tlf3Sr%2BAVyk1XBUk8bQSA%2F1l%2B2gCjESHvFui7Dk9aiGhgzbVJ4Knw0RV1QoNkkVYslgfudRcUlRyYOBaj2ZITZ3YHSC0AFFoUPm0Fz%2F9MftvOIAjkuNNEpSkfoEMfa7E8vdK8wWmHP1DOsqQ%2BgTJ6YOYZRJqZI7E50ZFK5vwNOadxIL3weJlgu7UocICafUiPNrxoz436I1DhBvIhjbeaXsQZl7BgBX%2BkZk7P344cr9g0MAcozMVpxY0z4ABHDfjTy7EVbF2892vulSKy8vdLvuZqhksoJs2rxymSlFp8AVBuNtstct0BieGGrlvPqKVBAUMC1WGVV4bp6G8lF32SgzCx%2BSMMLFyo8GOpoB6mSL%2BVchNE9l958aZKWqZD2FpwL%2B%2FLM1bxkYndncNT65%2F%2Bj5XvKMsGHR0nTszznCs5Szt42MZn8jZiJMEzc0Dr0PIgZrISmxZQX8Mao8I6t9oY7YYDXdCXd0IsEUS3HbKf4GpIdQvS28%2FE4X1t71zdDH9H8TokrTwepuiS5VR1Qwl%2Bn3%2ByFeIOQs3NRQuc8JK9J3vBWq2bx1mw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA5MF2VVMNGCYCIBEG%2F20220127%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220127T140341Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Signature=a9cec665acaab28e56fdb0e220c95697d555875a2b9822a0758046fa458d9187)

From the graph, we can see that before $\lambda = 2$, the bootstrap standard error converges to the Cramer-Rao lower bound as the sample size increases. The bootstrap standard error converges to 0 as the sample size increases. However, after $\lambda=2$ or greater, the Cramer-rao lower bound is reached, meaning that the MLE is **biased** because the Cramer-Rao lower bound is the minimum variance that can be achieved with an asymptotically unbiased estimator of $\lambda$. <sub><sup>or my derivation of MLE or implementations are wrong</sup></sub>

# Problem 3

For the last part of this problem set, I want you to think about **classification** models.  At their core, classification is a task that seeks to uncover a set of rules that determine whether an observation has or does not have a specific trait - Did the member of Congress cast a "Yes" vote for the bill? Will the customer buy a $2100 phone? Does the patient have cancer?  These are all questions that can addressed using classification models.

We're going to talk about a variety of approaches for building classification models.  However, all methods share a similar core strategy: given a set of predictors, what is a **rule set** that best predicts the observations true class?  For this problem, I want you to try to build a common-sense rules based model that predicts whether or not a patient has heart disease.

## Part 1 (15 pts.)

The data set `heartTrain.csv` contains information about 200 patients that were tested for heart disease.  `AHD` tells whether a patient was diagnosed with heart disease (`AHD` = 1) or not (`AHD` = 0).  There are also five predictors included - 1) `Age` (in years), 2) `Sex` (0 is female, 1 is male), 3) `RestBP` (the patient's resting blood pressure in mm/Hg), 4) `Chol` (the patient's serum cholesterol in mg/dl), and 5) `Slope` (a three category measure of heart responsiveness to exercise (a.k.a. the ST segment) - 1 means that the heart increased activity during exercise, 2 means that the heart activity remained constant, while 3 means that heart activity decreased during exercise).  

Using this data set, write a function called `heartDisease_predict` that takes in values for each of the five predictors and returns a prediction for whether or not the patient has a heart disease.  Then, use your predictive model to assess the **accuracy** of the predictions by comparing them to the real labels.  What proportion are correct?  What proportion are incorrect?

Your predictive model should only use if/else style rules (for example, if `Sex = 0` and `Rest BP < 140` then `AHD = 0`).  You can determine these rules by examining summary statistics, plots, running regression models, etc.  

You'll receive 10 points for the programming of your function and explanation of your approach.  You'll receive 2 more points for this problem if your model gets at least 60% of your predictions are correct, 3 more points pts. for at least 75%, and 5 more points for at least 85%.  If you come up with an approach that gets more than 95% correct, I'll add a bonus point to this assignment.  You can use any combination of predictors, transformations, and approaches to determine the rules set.


### Part 1 Solution
![part1](https://cleanshot-cloud-fra.s3.eu-central-1.amazonaws.com/media/27554/9G7HOKvhrTfcJrYAwXusTyCzUUGiBRL5Up2SQubc.jpeg?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJIMEYCIQC2gqvzt1OrwewPLqK3Z0dghrh7X2xC7k8bGRMUroTF1QIhAKSDlTqPbsS%2BdvHeuv%2BN9RLnT3pGyXeGH%2BGw%2B3h5RatpKqoCCL7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMOTE5NTE0NDkxNjc0Igw56Hq1jjQhz0tsNA4q%2FgGD71WYHHKMxjytER%2FKkQU5%2FlLUKtQllTQNg5SmNd5TV8pVyUSVprUX2L92YJk%2Fwa3Kg7fktPoiAG%2FTSaID4fs5%2FGIodD1aAcAmDydG9AjwNVUcFPN%2FdPhfDAGc8WCmiz3j0WOWkKWHjNBhXtnwa69frhyPUA4eHDCNlPriC8YhXxGyfzdZK44yGF%2F9S98OpaAmFKrLdOqKD1dKoqcAignArUxDCSftXUe9azJFaFv%2FTtRmm343230vV49PyM96K0PTe4ego9%2BCZ7lz%2FTtqasEF8I6oqSgVKYg5NJ1KzaaI8eW6pUvC5BAY%2FvAH96us7WzQQ%2BHmHDPumK3gQ2vMtzD2tsqPBjqZAZcXjrEnzVbA%2F9thLrXzNdUfGe7hQAE15lSnCLcgcQWs84rT03qin15577XuzajU3gvzv8aSeiV%2F7vcnjekYGehbMeampnjhNxq9UKjd4PO%2BKwr%2Bls3nySpx3EjhOuG5JkStn41VhRtN1gpLmaW8GdAuLyUOc%2Fc072U4VwQweq5%2BUcyEascsM47iLLlwEk6FXH8b6NxB6mQukQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA5MF2VVMNN4PTGAWV%2F20220127%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220127T140038Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Signature=c95aa2462b250829933e57ce79d31110966962825cf2c4e02bbdcc4efaa5e99b)

## Part 2 (5 pts.)

There is a second data set, `heartTest.csv`, that includes more observations from the original data set.  Using your version of `heartDisease_predict`, create predictions for this second data set.  What is the predictive accuracy for the test data set?  Is it better or worse than the accuracy for the training set?  Provide some intuition for this result.


### Part 2 Solution

![part2](https://cleanshot-cloud-fra.s3.eu-central-1.amazonaws.com/media/27554/Pzh44AH85bHa1h2EIjoXdhLISSrY9E7gkBVFB1Bj.jpeg?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCICd1GkIERmfqk0aYeI1vJ4QkeiKGojc6y2HtHLwUfvDvAiEA0G9SSqWa3pUTT1kmyc9NvllppOnO6pIHkHtl9wUtwKQqqgIIv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5MTk1MTQ0OTE2NzQiDNI78INMVNJE2kC7Hyr%2BAY8FYdx67t%2F%2BN9cvqgGgNEgUeTVLqMmPAPAiiCHTeXZuIcLJq4Y14qZPx5nnJPLXGge30XW%2BZ9hiOaw7MquxGt3nkqdOpzcPgOWWB2PdG6z0bDyBBF6d3nHBiRQGf41AvWByk87PdcaUuwc%2FqKf7Dgxin7WzvRVYV%2BLV4cUtKo3Jl35VlYQ6PYvhzicVYtoUZWYq7CjEhH5jAFvA778Ogrkz2lDqYehzjiLpigz%2FgrZtt2PxtkVgxEKisZrDribNY9TaIrooNOpDabEZld%2Bl%2BmocWpTAqbn1835i1IqSb6dQ7J8XaxJ1wd1PO0iMHyG5TUXJ%2BJnNWhe%2FD0c44a3uML3Ayo8GOpoB4tBECWGxlDMYDA16JFaAYGOS%2FECj3fR3l2aqjzJEeFxWy4Fp3om4Vj99av8iPZ4oZRPwoTDlL4IuG1or4urr42RsTQ2L%2Bke2DXv1oj2NSTrjUQTwjJAmEZBWIzlC6Xu1eZ6NqCJoFocLrYISBykDuFQxnPGOmAX1JwpOdd302LctoC669VgjVItMSBfG2MEkyB3dJ2Ev3eVi9g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA5MF2VVMNJDFG7NNU%2F20220127%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20220127T140222Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Signature=4caf495b0cf5a5344cc5ad6fc63f725e6acb03b619ba1b0ae372c46ddff047b1)
