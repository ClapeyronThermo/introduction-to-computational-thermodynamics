### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 23962934-2638-4788-9677-ae42245801ec
begin
	using Clapeyron: acentric_factor
	using Clapeyron, ForwardDiff, LinearAlgebra, NLsolve,  Optimization, OptimizationOptimJL # Numerical packages
	using LaTeXStrings, Plots # Display and plotting
	using HypertextLiteral
	using BenchmarkTools
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 30f48408-f16e-11ec-3d6b-650f1bf7f435
md"""
# Section 3.4 - The Flash Problem

The flash problem refers to knowing **when** a fluid will undergo a phase split, **how many phases** the fluid will split into, and the **composition** of each phase. We use these calculations all across chemical engineering, particularly in separation processes. For example, these are used when modelling flash drums, distillation columns and liquid-liquid extraction. They are also required when simulating petroleum reservoirs and other complex liquid flow.

## Key concepts

Before we begin, there are a few important ideas and equations we should mention.

### Chemical equilibrium 

At equilibrium, we have the **equivalence of chemical potential**, so that for a mixture with $C$ components and $P$ phases $(\alpha, \beta, \dots, P)$ we have an equivalence relation for every component _i_ in _C_:

$$\mu_i^\alpha = \mu_i^\beta = \dots = \mu_i^P$$

Identically, we have the **equivalent of fugacity**:

$$f_i^\alpha = f_i^\beta = \dots = f_i^P$$

We can express the compositions at vapour liquid equilibrium (VLE) using **K factors**. These represent the distribution of each component between phases.

$$K_i = \frac{y_i}{x_i}$$

Using the equality of fugacity, we can also express this in terms of the fugacity coefficient

$$\begin{align}
f_i^L &= x_i \varphi_i^L P\\
f_i^V &= y_i \varphi_i^V P\\
x_i \varphi_i^L \cancel{P} &= y_i \varphi_i^V \cancel{P}\\
K_i &= \frac{\varphi_i^L}{\varphi_i^V}
\end{align}$$

### The "trivial solution"

In flash problems we run the risk of converging to the so-called trivial solution. This is where each phase is identical, meaning the equality of chemical potential is inherently satisfied. In a situation where we know there should be a 2-phase region, it can be hard to avoid converging to this solution.
"""

# ╔═╡ 09dae921-9730-48a0-94b0-dd825d0ed919
md"""
## The Rachford-Rice equation

For the case where we have **composition independent** K values and a suspected 2-phase region, we can solve for the phase compositions and distribution using just material balances. 

$$\begin{align}
z_i F &= x_i L + y_i V\\
z_i &= x_i (1-\beta) + y_i \beta\\
\frac{z_i}{x_i} &= 1-\beta + K_i \beta\\
x_i &= \frac{z_i}{1+\beta(K_i - 1)}\\
y_i &= K_i x_i = \frac{K_iz_i}{1+\beta(K_i - 1)}
\end{align}$$

where $\beta$ is the *vapour fraction*, expressed as

$$\beta = \frac{L}{V}$$

Since mole fractions sum to one,

$$\sum x_i = \sum y_i = 1$$

we can subtract each equation from each other to obtain our objective function, the Rachford-Rice equation.

$$f(\beta) = \sum_i^{N_c} \frac{(K_i - 1)z_i}{1 + \beta(K_i - 1)} = 0$$

This formulation carries a few advantages over the other possible formulations. It **monotonic** and has easily obtainable analytical derivatives. When $K_i$ is known, it is then a univariate equation in $\beta$ easily solvable in a variety of ways. As the derivative is simple and the function monotonic, our preferred method in this course will be a **step limited Newton method**.
"""

# ╔═╡ f20d5217-cbe7-4878-ad45-f1f90881384b
let
	z = [0.4, 0.4, 0.2]
	K0 = [3.0, 0.25, 0.10]
	rr_plot(β) = sum(@. z*(K0 - 1) / (1 + β*(K0 - 1)))
	βmin = 1/(1 - maximum(K0))+1e-5
	βmax = 1/(1 - minimum(K0))-1e-5
	β_vec = range(βmin, βmax, length=1000)
	plot(title="The Rachford-Rice equation", xlabel="β", ylabel="f(β)", framestyle=:box, xlim=(βmin, βmax), ylim=(-4, 4), tick_direction=:out, grid=:off, legendfont=font(10))
	
	plot!(β_vec, rr_plot.(β_vec), label="", linewidth=2)
	vline!([0.0, 1.0], label="physical domain of β", linewidth=2)
	hline!([0.0], linecolor=:black, label="")
end

# ╔═╡ 2ec95442-1bd2-4133-864f-037bc5c35c9e
md"""
From inspecting our equation, we can see that the equation will diverge to infinity when 

$$\beta(K_i - 1) = -1$$

As we're summing across many $K_i$ values, the interval where the Rachford-Rice function is well behaved is described by

$$\beta \in \left(\frac{1}{1-K_\mathrm{max}}, \frac{1}{1-K_\mathrm{min}}\right)$$

Another key feature to notice is that this bracket often falls outside of the physical domain of $\beta$

$$\beta_{\mathrm{physical}} \in \left[0, 1\right]$$

We refer to the case where the solution to the Rachford-Rice equation falls outside of this physical domain as a **negative flash**. This implies that VLE does exist at the overall conditions but the material balance is not satisfied, so the mixture does not split into multiple phases.

Once we have the vapour fraction we can apply the material balances to calculate the compositions of our liquid and vapour phases.
"""

# ╔═╡ 4964a590-ef0e-4e2c-a90a-fe238d5766cf
md"""
### Task: Solving the Rachford-Rice equation

Finish the functions below to solve for the phase distribution at the given K and z values.

Use the Rachford-Rice equation as defined above, and the first derivative

$$\begin{gather}
f(\beta) = \sum_i^{N_\mathrm{C}} \frac{(K_i - 1)z_i}{1 + \beta(K_i - 1)} = 0\\
f'(\beta) = -\sum_i^{N_\mathrm{C}} \frac{z_i(K_i-1)^2}{(1+\beta(K_i - 1))^2}
\end{gather}$$

Also use the Newton method in 1D, remembering it is defined as

$$x_{n+1} = x_n - \frac{f(x)}{f^\prime(x)}$$

The step-limiting will be done by checking if 

$$β_\mathrm{new} ≥ β_\mathrm{max}$$
or

$$β_\mathrm{new} \leq β_\mathrm{min}$$

and if either of those cases is true, reduce the step size by half until $β_\mathrm{new}$ is within the domain.

$$d_\mathrm{new} = \frac{d}{2}$$

where

$$d_0 = \frac{f(x)}{f^′(x)}$$

Your function should accept a composition vector $z$ and a K-factor vector $K$, and return a scalar for the vapour fraction $\beta$.
"""

# ╔═╡ ad558065-7d69-411d-a033-da3d9047547c
"""
	rachford_rice(z, K, β)

Evaluates the Rachford-Rice objective function and returns a tuple (f, f′),
where f is the function value at the given point and f′ is the derivative with respect to β.
"""
function rachford_rice(z, K, β)
	inner_vec = @. (K - 1) / (1 + β*(K - 1))
	f = sum(z.*inner_vec)
	f′ = -sum(z.*inner_vec.^2)
	return (f, f′)
end

# ╔═╡ 4ae44687-2b39-4e92-93a3-e40b74415231
"""
	solve_β(z, K)

Solves the Rachford-Rice equation for the vapour fraction β using a step-limited Newton method.
"""
function solve_β(z, K)
	
	βmin = 1/(1-maximum(K))
	βmax = 1/(1-minimum(K))

	# Initial guess for β. 
	β = (βmin + βmax)/2
	δβ = 1.0
	i = 0
	itersmax = 100
	# While the change is greater than our tolerance
	while i < itersmax && abs(δβ) > 1e-7
		i += 1
		f, f′ = rachford_rice(z, K, β)
		# Calculate the newton step
		d = f/f′

		# Keep β inside the limits of f(β)
		step_ok = false
		while !step_ok
			βnew = β - d
			if βnew > βmax || βnew < βmin # Reject newton step
				# println("reducing step size (i=$i)")
				d = 0.5*d
			else
				# println("continuing (i=$i, d=$d)")
				step_ok = true
				δβ = βnew - β
				β = βnew
			end
		end
	end
	if i == itersmax
		@warn "failed to converge in $i iterations\n δΒ = $δβ\n β = $β\n βmin = $βmin\n βmax = $βmax"
	end
	return β
end

# ╔═╡ fc3805e2-5293-4cd0-ac25-48c520efb654
begin
	z_RR = [0.4, 0.4, 0.2]
	K_RR = [3.0, 0.25, 0.10]
	β_RR = solve_β(z_RR, K_RR)
end

# ╔═╡ 00ebf34a-dd75-48d4-834e-f6e0560df38d
md"""
## The Isothermal Flash

The procedure above set out how to solve for the phase distribution for the case where our K-factors are composition independent, but this approximation is only valid for very ideal species. To move to more general phase split calculations, we must account for this.

To do this we use the fact that K factors can be defined in two ways

$$\begin{gather}
K_i = \frac{y_i}{x_i}\\
K_i = \frac{\varphi_i^L(p,T,\mathbf{x})}{\varphi_i^V(p,T,\mathbf y)}
\end{gather}$$

allowing us to formulate the problem as

$$\mathbf K = f(\mathbf K)$$

which we solve for the **fixed points**. We can use many ways to solve for the fixed points, but the most simple is **subsequent substitution**. This is when we iterate using

$$\mathbf K_{n+1} = f(\mathbf K_n)$$

until the value of $\mathbf K$ converges to a fixed point.

Using subsequent substitution, this algorithm is broken up into 4 stages

```
0. Specify the state
1. Calculate initial guesses

While not converged
2. Solve Rachford-Rice equation for x⃗, y⃗
3. Calculate new K⃗

When converged
4. Calculate final β, x⃗, y⃗
```

Where stages 2 and 3 are the va
"""

# ╔═╡ 33d28a0e-31f8-40f4-9060-a35a4fe3ecf6
md"""
### 0. Specifying our state

Before we begin any calculations, we need to specify a few things. We need to define the composition of our feed, our thermodynamic model, and the pressure and temperature we're conducting the flash at. Here, we're using **Peng Robinson** to model a 3 component mixture.

$\begin{array}{lc}
\hline \text{Component} & \text {Mole fraction} \\ \hline
\text{Methane} & 0.4 \\
\text{Ethane} & 0.4 \\
\text{Hydrogen Sulfide} & 0.2 \\
\hline
\end{array}$

$\begin{array}{lc}
\hline \text{Specification} & \text {Value} \\ \hline
\text{Pressure} & 255\text{ K} \\
\text{Temperature} & 60\text{ bar} \\
\hline
\end{array}$
"""

# ╔═╡ aaac38e8-1d06-46ed-9607-a8e2fe1752e9
md"""
### 1. Initial guesses - the Wilson equation

$$\ln K_i = \ln \frac{P_{c,i}}{P_i} + 5.373(1+\omega_i)\left(1-\frac{T_{c,i}}{T}\right)$$

This is the same initialisation procedure we used for our **stability analysis** algorithm. Another possibility would be to use the result from stability analysis as an initial guess. That can be valuable as good initial guesses are particularly important when avoiding converging to the **trivial solution**.
"""

# ╔═╡ 156674ce-b49e-44b6-8182-7f8da0a394af
# Check if exercise
# if !@isdefined(K0)
# 	not_defined(:K0)
# else
# 	let
# 		try
# 			function Wilson_K_factor_test(pure_model, P, T)
# 			Tc, Pc, _ = crit_pure(pure_model)
# 			ω = acentric_factor(pure_model)
# 			K = exp(log(Pc/P) + 5.373*(1+ω)*(1-Tc/T))
# 			return K
# 			end
			
# 			Ksol = Wilson_K_factor_test.(pure_models, P, T)
# 			pure_test_model = PR(["methane"])
# 			P_test, T_test = (1e6, 190.0)
# 			K_test = Wilson_K_factor(pure_test_model, P_test, T_test)
# 			K_test_sol = 0.9
# 			# Check if function defined correctly and if K0 calculated correctly
# 			if K0 ≈ Ksol
# 				correct()
# 			elseif K_test == K_test_sol
# 				almost(md"Make you've changed `K0` correctly")
# 			else # If nothing correct
# 				keep_working(md"Make sure you've changed both the function `Wilson_K_factor` and `K0` correctly")
# 			end
# 		catch
# 			keep_working(md"Make sure you've changed both the function `Wilson_K_factor` and `K0` correctly")
# 		end
# 	end
# end

# ╔═╡ 44097395-f6b9-414b-8c5a-980c97feb553
md"""
### 2. Rachford-Rice equation

For this, we will use the solver we just developed to calculate 

$$β = f(\mathbf K, \mathbf z)$$

Then use the material balances to calculate the phase compositions

$$\begin{gather}
x_i = \frac{z_i}{1+\beta(K_i - 1)}\\
y_i = K_i x_i
\end{gather}$$
"""

# ╔═╡ 6f088d8b-a987-4fbd-a8d8-58c0acbf8aa0
md"""
### 3. Update K-factors

To calculate the next set of K-factors we use the expression

$$K_i = \frac{\varphi_i^L}{\varphi_i^V}$$

where $\varphi$ can be obtained using Clapeyron with 

```julia
fugacity_coefficient(model, p, T, z; phase = :unknown)
```

"""

# ╔═╡ 24b7f6f7-8204-4a31-aef9-c1d64b754fc3
md"""
### 4. Return values

We then apply the same procudure as in step 2 to calculate the final return values

$$(β,~\mathbf x,~\mathbf y)$$
"""

# ╔═╡ 30226c89-7c84-439b-9ed5-be6ca2d7aecf
md"""
### Flowchart
"""

# ╔═╡ 2385cc43-e041-4b49-bd52-6426216bf7a5
Resource("https://i.imgur.com/yN2voJp.jpeg", :height => 600)

# ╔═╡ e870a839-4c93-4693-8f1c-7b370082beec
md"""
### Implementation
Below there is a working implementation of the flash described above. Try changing some parameters and see how the phase split changes!
"""

# ╔═╡ c07e1d8b-113a-4382-af00-16332569fc8e
begin
	comps = ["methane", "ethane", "hydrogen sulfide"] # Define the components
	model = PR(comps) # Create our fluid model
	pure_models = split_model.(model) # Create a vector of models for each of our pure substances
	
	# Describe our state
	p = 55e5 # Pa
	T = 255.0 # K
	z = [0.4, 0.4, 0.2]
end;

# ╔═╡ 5fb5e842-b684-47b9-a60e-deedc485914c
"""
	Wilson_K_factor(pure_model, p, T)

Returns the K-factor predicted by the Wilson correlation for a pure component at a given pressure and temperature
"""
function Wilson_K_factor(pure_model, p, T)
	Tc, pc, vc = crit_pure(pure_model)
	ω = acentric_factor(pure_model)
	
	# Complete the expression for calculating K-factors
	K = exp(log(pc/p) + 5.373*(1+ω)*(1-Tc/T))
	
	return K
end

# ╔═╡ 8fdc31ec-178a-4ffd-b029-5590f192332f
"""
	update_K_factors(model, p, T, x, y)
returns a vector of K-factors
"""
function update_K_factors(model, p, T, x, y)
	φᴸ = fugacity_coefficient(model, p, T, x; phase=:liquid)
	φⱽ = fugacity_coefficient(model, p, T, y; phase=:vapour)
	K = φᴸ./φⱽ
	return K
end

# ╔═╡ cec97051-aec3-4e17-95eb-90d19b947c37
K0 = Wilson_K_factor.(pure_models, p, T)

# ╔═╡ 90baae39-478b-493c-ac23-4fadca9c3698
begin
	try
		@htl("""
		<table>
		  <tr>
		    <th>Component</th>
		    <th>K-factor</th>
		  </tr>
		  <tr>
		    <td>$(comps[1])</td>
		    <td>$(round(K0[1]; digits=4))</td>
		  </tr>
		  <tr>
		    <td>$(comps[2])</td>
		    <td>$(round(K0[2]; digits=4))</td>
		  </tr>
		  <tr>
		    <td>$(comps[3])</td>
		    <td>$(round(K0[3]; digits=4))</td>
		  </tr>
		</table>
		""")
	catch
		@htl("""
		<table>
		  <tr>
		    <th>Component</th>
		    <th>K</th>
		  </tr>
		  <tr>
		    <td>$(comps[1])</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>$(comps[2])</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>$(comps[3])</td>
		    <td>?</td>
		  </tr>
		</table>
		""")
	end
end

# ╔═╡ 1453647c-213c-4627-a5b1-ffc1f95204a1
function solve_flash(model, p, T, z, K0; maxiters=100, abstol=1e-7)
	iters = 0
	K_norm = 1.0
	K = K0
	while K_norm > abstol && iters < maxiters
		iters += 1

		β = solve_β(z, K)
		x = @. z/(1 + β*(K - 1))
		y = @. K*x

		Knew = update_K_factors(model, p, T, x, y)
		K_norm = norm(Knew .- K)
		K = Knew
	end

	# Make sure to issue a warning if the flash failed to converge!
	iters == maxiters && @warn "did not converge in $iters iterations"

	β = solve_β(z, K)
	x = @. z/(1 + β*(K - 1))
	y = @. K*x
	return (β, x, y)
end

# ╔═╡ 96476f2f-a791-4ea6-9f81-8a17a00d6c1f
β_flash, x_flash, y_flash = solve_flash(model, p, T, z, K0)

# ╔═╡ 7b4e20ce-815a-46c7-bcc5-7b6531542049
let
	try
		@htl("""
		<table>
		  <tr>
		    <th rowspan="2">Component</th>
		    <th colspan="2">Mole fraction</th>
		  </tr>
		  <tr>
			<th>Liquid</th>
			<th>Vapour</th>
		  </tr>
		  <tr>
		    <td>$(comps[1])</td>
		    <td>$(round(x_flash[1]; digits=4))</td>
		    <td>$(round(y_flash[1]; digits=4))</td>
		  </tr>
		  <tr>
		    <td>$(comps[2])</td>
		    <td>$(round(x_flash[2]; digits=4))</td>
		    <td>$(round(y_flash[2]; digits=4))</td>
		  </tr>
		  <tr>
		    <td>$(comps[3])</td>
		    <td>$(round(x_flash[3]; digits=4))</td>
		    <td>$(round(y_flash[3]; digits=4))</td>
		  </tr>
		</table>
		<center>
		β = $(round(β_flash; digits=4))
		</center>
		""")
	catch
		@htl("""
		<table>
		  <tr>
		    <th rowspan="2">Component</th>
		    <th colspan="2">Mole fraction</th>
		  </tr>
		  <tr>
			<th>Liquid</th>
			<th>Vapour</th>
		  </tr>
		  <tr>
		    <td>$(comps[1])</td>
		    <td>?</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>$(comps[2])</td>
		    <td>?</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>$(comps[3])</td>
		    <td>?</td>
		    <td>?</td>
		  </tr>
		</table>
		<center>
		β = ?
		</center>
		""")
	end
end

# ╔═╡ 001c511c-34b9-4a25-b901-1b98a3eaea9b
md"""
## Convergence of the two-phase flash

Subsequent substitution converges very quickly for situations where the state is far from the critical point and when the K-factors are weakly dependent on composition. When we fall outside of these cases, this method can take hundreds of iterations to converge.

To visualise this, we can keep track of our convergence measure (the Euclidian norm of change in $\mathbf K$) and plot this against the number of iterations.

Here we're looking at three different numerical methods
1. Successive substitution
2. Accelerated successive substitution [^1]
3. Newton's method

In the figure below we show the convergence speed of three different iteration procedures for the flash calculation. We see the Newton algorithm converges very quickly, followed by the accelerated successive substitution, then finally the standard successive substitution procedure we wrote earlier.

However, before analysing their real-world performance, we should have an idea of how they may be expected to perform.

Successive substitution has **linear** convergence. This means that each successive error is approximately proportional, 

$$||x_{i+1} - x^*|| \leq c ||x_i - x^*||$$

where $x^*$ represents the final value, and $x_{i}, x_{i+1}$ represent iterations.

Accelerated successive substitution has **superlinear** convergence. This is when our convergence is faster than linear, but not yet quadratic. The acceleration procedure uses the last _m_ iterations, in this case 5, to extrapolate each step and converge faster.

Finally, Newton's method has **quadratic** convergence. Here, the error within each successive iteration obeys

$$||x_{i+1} - x^*|| \leq c \left(||x_i - x^*||\right)^2~.$$
"""

# ╔═╡ b04743cd-bdc8-46a7-be77-8b0e116fae3a
function rootfinding_obj_func!(model, F, lnK, p, T, z, x, y)
    K = exp10.(lnK)
    any(isnan.(K)) && @error ("K = $K\n F = $F")

    β = Clapeyron.rr_vle_vapor_fraction(K, z)
    # Clapeyron.rr_flash_liquid!(x, K, z, β)
	x = @. z/(1.0 + β*(K - 1.0))
	x = normalize(x, 1)
    y = @. K * x
    Knew = update_K_factors(model, p, T, x, y)

    F .= log10.(Knew) .- lnK
end

# ╔═╡ 734dd8c9-cf26-4cf8-baf2-63cf97c34c2b
begin
	f!(F, K) = rootfinding_obj_func!(model, F, K, p, T, z, zeros(length(z)), zeros(length(z)))
	
	res_vec = []

	res = nlsolve(f!, log10.(K0); store_trace=true, method=:anderson, m=0)
	push!(res_vec, ["successive substitution", res])

	res = nlsolve(f!, log10.(K0); store_trace=true, method=:anderson, m=5)
	push!(res_vec, ["accelerated successive substitution", res])

	res = nlsolve(f!, log10.(K0); autodiff=:forward, store_trace=true, method=:newton)
	push!(res_vec, ["newton", res])
end; 

# ╔═╡ 457b66ff-7972-49b8-b57a-4b4fc5f0ad06
let
	p = plot(title="Convergence characteristics of flash calculations", xlabel="iteration", ylabel="error", framestyle=:box, tick_direction=:out, grid=:off, yaxis=:log, yticks=exp10.(range(-11, 0)), xlim=(0, 25), legendfont=font(10))
	
	for (method, res) in res_vec
		iter = [x.iteration for x in res.trace.states]
		method == "newton" && map!(x -> x+1, iter, iter)
		residual = [x.fnorm for x in res.trace.states]
		plot!(p, iter, residual, linewidth=2, label=string(method))
	end
	p
end

# ╔═╡ 6b9fdb5c-db31-483e-8329-0ebebb9ba40f
md"""
### Benchmarks
"""

# ╔═╡ 0b08a3d1-d857-4387-8c61-da9899edeae2
md"""
#### Successive substitution
"""

# ╔═╡ 539b985c-2973-41f3-9b8c-9879184d8cf3
let
	test = @benchmarkable nlsolve(f!, log10.(K0); store_trace=true, method=:anderson, m=0)
	run(test, samples=1)
	run(test, samples=10000)
end

# ╔═╡ 8f4cdabd-b424-4f16-8129-d315b7f77aad
md"""
#### Accelerated successive substitution [^1]
"""

# ╔═╡ 3cdba133-68cb-4c3e-bf30-e4b235a321a1
let
	test = @benchmarkable nlsolve(f!, log10.(K0); store_trace=true, method=:anderson, m=5)
	run(test, samples=10000)
end

# ╔═╡ c2fb190a-0b65-4356-bf89-62f776ea7fe8
md"""
#### Newton
"""

# ╔═╡ 48d547f4-bb0a-4c85-9ef4-4013bec76752
let
	test = @benchmarkable nlsolve(f!, log10.(K0); autodiff=:forward, method=:newton)
	run(test, samples=10000)
end

# ╔═╡ 32870a35-1276-4d91-aa55-64f6256b4188
md"""
From the error-iteration graph above we can see that our real-world convergence represents what we expected. However, the benchmarks then show that Newton method in this case performs worse than even successive substitution!

This occurs because of the increased complexity of Newton's method. Requiring a full Jacobian for each iteration, as well as the subsequent linear solve, there is considerably more computational cost for each iteration.

Attempting to get the "best of both worlds", Michelsen proposed a combination algorithm, leveraging the advantages of both accelerated successive substitution and the Newton method. As we can see accelerated successive substitution converges very quickly for most situations, we perform up to 15 iterations of this method, then switch to a faster converging, but slower to evalute, method like Newton.
"""

# ╔═╡ e715a1b3-c11b-4d01-96d3-af3b64dee96c
# let
# 	model = PR(["methane","butane","isobutane","pentane"])

# 	x = [0.96611, 0.01475, 0.01527, 0.00385]
# 	Tc, pc, Vc = crit_mix(model, x)
	
# 	Tcrit = Tc
# 	Tmax = 265.5
	
# 	T1 = range(150, Tcrit, length=400)
# 	T2 = range(200, Tmax, length=400)
# 	T3 = range(Tmax, 215, length=400)
	
# 	# Preallocate arrays
# 	p1 = zeros(400)
# 	p2 = zeros(400)
# 	p3 = zeros(400)
	
# 	v1 = zeros(6)
# 	v2 = zeros(6)
	
# 	for (i, (T1ᵢ, T2ᵢ)) in enumerate(zip(T1, T2))
# 	    if i == 1
# 	        bub1 = bubble_pressure(model, T1ᵢ, x)
# 	        bub2 = dew_pressure(model, T2ᵢ, x)
# 	    else
# 	        bub1 = bubble_pressure(model, T1ᵢ, x; v0=v1)
# 	        bub2 = dew_pressure(model, T2ᵢ, x; v0=v2)
# 	    end
# 	    p1[i] = bub1[1]
# 	    p2[i] = bub2[1]
# 	    v1 .= [log10(bub1[2]), log10(bub1[3]), bub1[4][1], bub1[4][2], bub1[4][3], bub1[4][4]]
# 	    v2 .= [log10(bub2[2]), log10(bub2[3]), bub2[4][1], bub2[4][2], bub2[4][3], bub2[4][4]]
# 	end
	
# 	v3 = v2
# 	v3[2] = 1.1*v3[2]
	
# 	for (i, T3ᵢ) in enumerate(T3)
# 	    bub3 = dew_pressure(model, T3ᵢ, x; v0=v3)
# 	    p3[i] = bub3[1]
# 	    v3 .= [log10(bub3[2]), log10(bub3[3]), bub3[4][1], bub3[4][2], bub3[4][3], bub3[4][4]]
# 	end
	
# 	T_plot = vcat(T1, reverse(T3), reverse(T2));
# 	p_plot = vcat(p1, reverse(p3), reverse(p2));

# 	iters_p = Vector{Float64}()
# 	iters_T = Vector{Float64}()
# 	iters_vec = Vector{Float64}()
	
# 	# stable_bool = false
# 	for p in range(2e6, 7e6, 10)
# 	    for T in range(190, 260.0, 10)
# 	        stable_bool, _ = check_chemical_stability(model, p, T, x)
# 	        if ~stable_bool #&& (abs((T - Tc)/Tc) > 0.1 || abs((p - pc)/pc) > 0.1)
# 	            try
# 		            iters = flash_iterations(model, p, T, x)
# 					# if iters < 50
# 					push!(iters_p, p)
# 					push!(iters_T, T)
# 					push!(iters_vec, iters)
# 					# end
# 	            catch e
# 					if isa(e, NLsolve.IsFiniteException)
# 						continue
# 					else
# 						@show typeof(e)
# 						rethrow()
# 					end
# 	            end
# 	        end
# 	    end
# 	end
	
# 	plot(title="pT isopleth", xlabel="T / K", ylabel="p / MPa", framestyle=:box, tick_direction=:out, grid=:off, right_margin = 4Plots.mm, thickness_scaling=1.2)

# 	scatter!(iters_T, iters_p./1e6, zcolor=iters_vec, colorbar_title="\nflash iterations", label="", c = :plasma)
	
# 	plot!(T_plot, p_plot./1e6, linewidth=3, label="binodal", color=1)
	
# 	scatter!([Tc], [pc/1e6], markersize=7, color=:red, label="critical point")
# 	plot!(legend=:topleft)
# end

# ╔═╡ ea5bbbc9-6d08-4f31-b657-dfb884f181c1
function flash_iterations(model, p, T, z; SSiters=5, maxiters=70, abstol=1e-5)
	K0 = Clapeyron.wilson_k_values(model, p, T)
	x = zeros(length(z))
	y = zeros(length(z))
	β = 0.5
	f!(F, lnK) = rootfinding_obj_func!(model, F, lnK, p, T, z, x, y)
	res = nlsolve(f!, log10.(K0); method=:anderson, m=0, iterations=SSiters, ftol=abstol, xtol=abstol)
	iters = res.iterations
	if iters == SSiters
		res = nlsolve(f!, res.zero; method=:newton, iterations=maxiters-SSiters, ftol=abstol, xtol=abstol)
		iters += res.iterations
	end
	return iters
end

# ╔═╡ 4acb1393-030f-4cab-a765-f8de5a75893b
html"<br><br><br><br><br><br><br><br><br><br><br><br>"

# ╔═╡ 5e64af2a-b467-41f7-98f8-83d49e210a32
md"""
# Footnotes
[^1]: Here we use [Anderson acceleration](https://en.wikipedia.org/wiki/Anderson_acceleration) to speed up the convergence rate of our fixpoint iteration. This is slightly different to the methods usually used in literature, e.g. by Michelsen. There you could expect to see some form of eigenvalue extrapolation, either dominant eigenvalue method (DEM) or general dominant eigenvalue method (GDEM). Another "accelerated successive substitution" method is described by _Risnes et al_, and is described [here](https://www.e-education.psu.edu/png520/m17_p6.html).
"""

# ╔═╡ d0b2f6bb-7539-4dda-adc9-acc2ce9cca4a
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]));

# ╔═╡ 7b75b351-6490-4c33-ab77-f42f6e1453d9
hint(title, text) = Markdown.MD(Markdown.Admonition("hint", title, [text]));

# ╔═╡ d6778a9d-2455-434d-b459-27195ac7a59f
hint(md"""
Consider using the broadcasting macro @. to simplify your expression in rachford_rice
""")

# ╔═╡ c2e866df-a46b-4ca0-8b3f-e7fa2e7b4143
hint("Hint 1", md"""
Calculate your Newton step using 

$$d = \frac{f}{f^′}$$

You can write ′ by typing \prime then pressing tab.
""")

# ╔═╡ 8fe83aab-d193-4a28-a763-6420abcbb176
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]));

# ╔═╡ 059d43c4-e94a-442a-a89a-20d83d20180b
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]));

# ╔═╡ 82fbda65-888a-4be5-b515-cf63fe5ca305
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]));

# ╔═╡ 575f4fe6-36dd-4e82-9c32-d84959f66477
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));

# ╔═╡ 80732074-1d94-45ed-8e5e-ea92d3985a1c
if !@isdefined(β_RR)
	not_defined(:β_RR)
else
	let
		try
			correct()
			# # Check if function defined correctly and if K0 calculated correctly
			# if K0 ≈ Ksol
			# 	correct()
			# elseif K_test == K_test_sol
			# 	almost(md"Make you've changed `K0` correctly")
			# else # If nothing correct
			# 	keep_working(md"Make sure you've changed both the function `Wilson_K_factor` and `K0` correctly")
			# end
		catch
			keep_working()
		end
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Clapeyron = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.3.1"
Clapeyron = "~0.3.7"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
NLsolve = "~4.5.1"
Optimization = "~3.8.0"
OptimizationOptimJL = "~0.1.2"
Plots = "~1.31.4"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "8d9e48436c5589fbd51ae8c8165a299a219188c0"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.15"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.BlackBoxOptim]]
deps = ["CPUTime", "Compat", "Distributed", "Distributions", "HTTP", "JSON", "LinearAlgebra", "Printf", "Random", "SpatialIndexing", "StatsBase"]
git-tree-sha1 = "41e347c63757dde7d22b2665b4efe835571983d4"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clapeyron]]
deps = ["BlackBoxOptim", "CSV", "DiffResults", "FillArrays", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "NLSolvers", "PackedVectorsOfVectors", "PositiveFactorizations", "Roots", "Scratch", "SparseArrays", "StaticArrays", "Tables", "ThermoState", "UUIDs", "Unitful"]
git-tree-sha1 = "1ebd358d7c650500aab466d28c55b45a020553ae"
uuid = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
version = "0.3.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSolve]]
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "aafa0665e3db0d3d0890cdc8191ea03dc279b042"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.66"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "cb8c5f0074153ace28ce5100714df4378ad885e0"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.14.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "d88b17a38322e153c519f5a9ed8d91e9baa03d8f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "037a1ca47e8a5989cc07d19729567bb71bfabd0c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "c8ab731c9127cd931c93221f65d6a1008dad7256"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7c88f63f9f0eb5929f15695af9a4d7d3ed278a91"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.16"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "9f4f5a42de3300439cb8300236925670f844a555"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NLSolvers]]
deps = ["IterativeSolvers", "LinearAlgebra", "PositiveFactorizations", "Printf", "Statistics"]
git-tree-sha1 = "93d2f4b482aad8e90af7e332b705572d2c104191"
uuid = "337daf1e-9722-11e9-073e-8b9effe078ba"
version = "0.2.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.Optimization]]
deps = ["ArrayInterfaceCore", "ConsoleProgressMonitor", "DiffResults", "DocStringExtensions", "Logging", "LoggingExtras", "Pkg", "Printf", "ProgressLogging", "Reexport", "Requires", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "34e947f0cdb8172f25533279d6abc28d03b29584"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "3.8.0"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "Optimization", "Reexport", "SparseArrays"]
git-tree-sha1 = "76ac41f9e82ba98a600d8a380532adef76b27e15"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.1.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PackedVectorsOfVectors]]
git-tree-sha1 = "78a46960967e9e37f81dbf7f61b45b0159637afe"
uuid = "7713531c-48ef-4bdd-9821-9ff7a8736089"
version = "0.1.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0a0da27969e8b6b2ee67c112dcf7001a659049a0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2690681814016887462cf5ac37102b51cd9ec781"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ce7584604489e537b2ab84ed92b4107d03377f0"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.31.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "50f945fb7d7fdece03bbc76ff1ab96170f64a892"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "a5d629df514350ff95fa9cc75c43296bebfdf427"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.44.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "77172cadd2fdfa0c84c87e3a01215a4ca7723310"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.0.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpatialIndexing]]
git-tree-sha1 = "fb7041e6bd266266fa7cdeb80427579e55275e4f"
uuid = "d4ead438-fe20-5cc5-a293-4fd39a41b74c"
version = "0.1.3"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "0005d75f43ff23688914536c5e9d5ac94f8077f7"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.20"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThermoState]]
deps = ["Unitful"]
git-tree-sha1 = "9a89a06e84165557c2b720cd2479053cfcc4f74f"
uuid = "e7b6519d-fdf7-4a33-b544-2b37a9c1234a"
version = "0.5.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─23962934-2638-4788-9677-ae42245801ec
# ╟─30f48408-f16e-11ec-3d6b-650f1bf7f435
# ╟─09dae921-9730-48a0-94b0-dd825d0ed919
# ╟─f20d5217-cbe7-4878-ad45-f1f90881384b
# ╟─2ec95442-1bd2-4133-864f-037bc5c35c9e
# ╟─4964a590-ef0e-4e2c-a90a-fe238d5766cf
# ╠═ad558065-7d69-411d-a033-da3d9047547c
# ╟─d6778a9d-2455-434d-b459-27195ac7a59f
# ╠═4ae44687-2b39-4e92-93a3-e40b74415231
# ╠═fc3805e2-5293-4cd0-ac25-48c520efb654
# ╟─80732074-1d94-45ed-8e5e-ea92d3985a1c
# ╟─c2e866df-a46b-4ca0-8b3f-e7fa2e7b4143
# ╟─00ebf34a-dd75-48d4-834e-f6e0560df38d
# ╟─33d28a0e-31f8-40f4-9060-a35a4fe3ecf6
# ╟─aaac38e8-1d06-46ed-9607-a8e2fe1752e9
# ╟─90baae39-478b-493c-ac23-4fadca9c3698
# ╟─156674ce-b49e-44b6-8182-7f8da0a394af
# ╟─44097395-f6b9-414b-8c5a-980c97feb553
# ╟─6f088d8b-a987-4fbd-a8d8-58c0acbf8aa0
# ╟─24b7f6f7-8204-4a31-aef9-c1d64b754fc3
# ╟─30226c89-7c84-439b-9ed5-be6ca2d7aecf
# ╟─2385cc43-e041-4b49-bd52-6426216bf7a5
# ╟─e870a839-4c93-4693-8f1c-7b370082beec
# ╠═c07e1d8b-113a-4382-af00-16332569fc8e
# ╠═5fb5e842-b684-47b9-a60e-deedc485914c
# ╠═8fdc31ec-178a-4ffd-b029-5590f192332f
# ╠═cec97051-aec3-4e17-95eb-90d19b947c37
# ╠═1453647c-213c-4627-a5b1-ffc1f95204a1
# ╠═96476f2f-a791-4ea6-9f81-8a17a00d6c1f
# ╟─7b4e20ce-815a-46c7-bcc5-7b6531542049
# ╟─001c511c-34b9-4a25-b901-1b98a3eaea9b
# ╟─b04743cd-bdc8-46a7-be77-8b0e116fae3a
# ╟─734dd8c9-cf26-4cf8-baf2-63cf97c34c2b
# ╟─457b66ff-7972-49b8-b57a-4b4fc5f0ad06
# ╟─6b9fdb5c-db31-483e-8329-0ebebb9ba40f
# ╟─0b08a3d1-d857-4387-8c61-da9899edeae2
# ╟─539b985c-2973-41f3-9b8c-9879184d8cf3
# ╟─8f4cdabd-b424-4f16-8129-d315b7f77aad
# ╟─3cdba133-68cb-4c3e-bf30-e4b235a321a1
# ╟─c2fb190a-0b65-4356-bf89-62f776ea7fe8
# ╟─48d547f4-bb0a-4c85-9ef4-4013bec76752
# ╟─32870a35-1276-4d91-aa55-64f6256b4188
# ╟─e715a1b3-c11b-4d01-96d3-af3b64dee96c
# ╟─ea5bbbc9-6d08-4f31-b657-dfb884f181c1
# ╟─4acb1393-030f-4cab-a765-f8de5a75893b
# ╟─5e64af2a-b467-41f7-98f8-83d49e210a32
# ╟─d0b2f6bb-7539-4dda-adc9-acc2ce9cca4a
# ╟─7b75b351-6490-4c33-ab77-f42f6e1453d9
# ╟─8fe83aab-d193-4a28-a763-6420abcbb176
# ╟─059d43c4-e94a-442a-a89a-20d83d20180b
# ╟─82fbda65-888a-4be5-b515-cf63fe5ca305
# ╟─575f4fe6-36dd-4e82-9c32-d84959f66477
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
