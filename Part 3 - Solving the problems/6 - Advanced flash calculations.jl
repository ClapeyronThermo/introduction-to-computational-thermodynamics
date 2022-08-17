### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bb890ff2-fd86-11ec-11e6-6f26bc733df1


# ╔═╡ dcd9cc3a-c55c-4b31-9ac6-8d0e8106fd90
md"""
### Section 3.5
# Advanced Flash Calculations

Now you've seen and implemented a 2-phase $pT$ flash, the natural question to ask is what about other specifications? Just as the $pT$ flash can be expressed as a Gibbs minimisation problem, each specification also minimises a given state function.

$\begin{array}{ccl}
\hline \text{Specification} & \text {State function} & \text{Example use} \\ \hline
p,T & G & \text{Flash drum}\\
p,H & -S & \text{Adiabatic expansion}\\
p,S & H & \text{Reversible expansion}\\
V,T & A & \text{}\\
V,U & -S & \text{Dynamic fluid simulation}\\
\hline
\end{array}$

While $pT$ flash algorithms are the most common, $VT$ based algorithms are also relatively standard. This 


"""

# ╔═╡ 98e6c49b-1b8a-4b30-afcc-0c820ae82cf2
md"""
## $pH$ flash formulation

The constrained minimisation problem for a $pH$ flash would be written as 

> $$\min(-S(T, P_\mathrm{spec}, \mathbf v, \mathbf l))$$
> 
> subject to:
> 
> $$\begin{gather}
> H(T, P_\mathrm{spec}, \mathbf v, \mathbf l) = H_\mathrm{spec}\\
> \mathbf v + \mathbf l = \mathbf z
> \end{gather}$$

The material balance constraint can be eliminated with

$$\mathbf l = \mathbf z - \mathbf v$$

To give us

> $$\min(-S(T, P_\mathrm{spec}, \mathbf v, \mathbf z - \mathbf v))$$
> 
> subject to:
> 
> $$\begin{gather}
> H(T, P_\mathrm{spec}, \mathbf v, \mathbf z - \mathbf v) = H_\mathrm{spec}\\
> \end{gather}$$

But this still leaves an unpleasant constraint. To avoid this entirely, we reformulate the objective function to a objective, called a "Q-function". This allows us to use any flash specification with a core of either a $pT$ or $VT$ flash.

The Q-function for a $pH$ flash is written as

$$Q(T,P_\mathrm{spec},\mathbf v, \mathbf z - \mathbf v) = \frac{1}{T}(G-H_\mathbf{spec})$$

with gradient vectors

$$\begin{gather}
\frac{\partial Q}{\partial v_i} = \frac{1}{T}\frac{\partial G}{\partial v_i} = \frac{1}{T}(\mu_i^V - \mu_i^L)\\
\frac{\partial Q}{\partial (1/T)} = \frac{\partial(G/T)}{\partial(1/T)} - H_\mathrm{spec} = H - H_\mathrm{spec}
\end{gather}$$

At equilibrium the gradient is 0, 

## All unconstrained flash formulations

Every flash specification can be expressed in this way with a Q-function. This allows for very flexible implementation of flashes with a core of just one $pT$ or $VT$ algorithm.

$$\begin{array}{cc}
\hline \text{Specification} & \text{Gibbs-based Q-function} & \text{Helmholtz-based Q-function} \\ \hline
p,T & G_\mathrm{min} & A_\mathrm{min} + VP_\mathrm{spec}\\
p,H & \frac{1}{T}(G_\mathrm{min}-H_{\mathrm{spec}}) & \frac{1}{T}(A_\mathrm{min}+VP_\mathrm{spec}-H_\mathrm{spec})\\
p,S & G_\mathrm{min}+TS_\mathrm{spec} & A_\mathrm{min} + TS_\mathrm{spec} + VP_\mathrm{spec}\\
T,V & G_\mathrm{min}-PV_\mathrm{spec} & A_\mathrm{min}\\
U,V & \frac{1}{T}(G_\mathrm{min}-U_\mathrm{spec} - PV_\mathrm{spec}) & \frac{1}{T}(A_\mathrm{min}-U_\mathrm{spec})\\
S,V & G_\mathrm{min}+TS_\mathrm{spec}-PV_\mathrm{spec} & A_\mathrm{min} + TS_\mathrm{spec}\\
\hline
\end{array}$$

The simplest way to implement this is as a **bilevel optimisation** problem, meaning an outer problem maximising the Q-function, where each iteration requires the solution to a $VT$ or $PT$ flash problem minimising $A$ or $G$ respectively. This corresponds to a saddle point of the Q-function, negative curvature in $T$ or $P$ and positive curvature in the composition.

It's interesting to note that the Helmholtz-based Q-function is potentially valuable for SAFT type equations, which are formulated in terms of volume and temperature.
"""

# ╔═╡ 09ea9e95-891e-465b-b2c1-7a8869fdb7f4
md"""
## Multiphase flash calculations

Up until now we have primarily looked at vapour-liquid equilibrium (VLE), but other forms of equilibrium are common and can bring their own challenges.

### LLE

In liquid-liquid equilibria initial guesses become hard. The vapourlike guess from the Wilson correlation is very likely to be significantly incorrect, and convergence to a trivial solution can become hard to avoid.

### SLE

Solids have a defined fugacity, but aren't usually described by typical equations of state.

### VLLE

If we have known VLLE we can anticipate this and create initial guesses targetting it.

### Unknown no. phases

When we do not know the number of phases before beginning our flash calculation the procedure can become significantly more complicated. The algorithm now needs to be able to both add and remove phases, as well as ensure that the final reported phase distribution is stable.

#### The Michelsen algorithm

The most well known algorithm for phase equilibria calculations, other than the basic Rachford-Rice based iteration you saw in the previous section, is likely the Michelsen multiphase algorithm. In this there are two key modifications:

1) Operating with a modified Rachford-Rice objective function for equilibrium calculations with more than two phases.

2) We perform stability analysis on each phase in the resulting phase distribution, and if any turn out to be unstable we add another phase and perform the flash calculation again.

#### Helmholtz Energy Lagrangian Dual

This is an advanced flash algorithm that uses random numbers and global optimisation techniques to perform flashes across unknown numbers of phases without having to specify initial guesses. One of the key features of this algorithm is the ability to map a $pT$ specification to a Helmholtz based objective function. As we discussed earlier, this make it powerful for use with more complex equations of state that are generally formulated in terms of volume by avoiding potential numerical instability caused by an inner nonlinear solver.

While this algorithm is very safe, it does come at significant computational cost and would not be the preferable default choice for a situation where a large number of flashes are required, like in a process simulator.

## Chemical equilibrium calculations

Calculating the position of chemical equilibrium in a reaction network is a surprisingly small modification to our existing algorithms. As we are only considering the position of equilibrium, rather than any reaction rates, it just becomes a case of adding material balance constraints. These can then be solved with any optimisation algorithm that supports constraints, for example interior-point Newton or BFGS-B.

For example, considering a reaction network

$$\begin{gather}
1)\quad\ce{A + B <=> C}\\
2)\quad\ce{D + 2E <=> 3F + G}
\end{gather}$$

we can define the extent vector $\boldsymbol \xi$, allowing our optimisation problem to be posed as

> $$\min G(p,T,\mathbf n_\mathrm{0} - \boldsymbol \xi)$$
>
> subject to:
>
> $$\begin{gather}
> 0 \leq \xi_1 \leq \min \{n_\mathrm{A0},n_\mathrm{B0}\}\\
> 0 \leq \xi_2 \leq \min \{n_\mathrm{D0}, 2n_\mathrm{E0}\}
> \end{gather}$$

However, care must be taken that **every reaction is linearly independent**. For example, this means that using our reaction network above we could not have a third reaction

$$\ce{A + B + D + 2E <=> 3F + G}$$

as this would lead to an underdefined system. We can check this by describing our reaction network using a matrix, and checking the rank of that matrix.
"""

# ╔═╡ 388a0adc-4ad9-4e25-a609-ceb451190bbb


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─bb890ff2-fd86-11ec-11e6-6f26bc733df1
# ╟─dcd9cc3a-c55c-4b31-9ac6-8d0e8106fd90
# ╟─98e6c49b-1b8a-4b30-afcc-0c820ae82cf2
# ╟─09ea9e95-891e-465b-b2c1-7a8869fdb7f4
# ╠═388a0adc-4ad9-4e25-a609-ceb451190bbb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
