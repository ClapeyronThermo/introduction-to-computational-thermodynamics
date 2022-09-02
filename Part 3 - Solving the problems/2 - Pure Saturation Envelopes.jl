### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 58b0aaa8-f47a-11ec-1113-9933151340be
begin
	using Clapeyron: VT_chemical_potential, ABCubicModel, SAFTModel, p_scale, T_scale, R̄, N_A
	const R = R̄;
	
	using Clapeyron, ForwardDiff, Roots, NLsolve, LinearAlgebra # Numerical packages
	using LaTeXStrings, Plots, ShortCodes, Printf # Display and plotting
	using HypertextLiteral
	import PlotlyJS
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ c4b5bc07-7f17-47d2-a3e6-945813e29b35
md"""
# Section 3.2 - Saturation solvers for pure substances

According to the Gibbs phase rule, we have one degree of freedom along the saturation curves for a pure substance. This means that we should be able to specify either pressure or temperature and determine the corresponding saturated point.

This can be solved in two ways, either as an optimisation problem formulated in just pressure or chemical potential, or a root-finding problem. Generally root-finding problems are easier to solve numerically, so most often this is the formulation that is preferable.

## Specification of phase equilibrium

We know that at equilibrium we have the equalities of temperature, pressure, and chemical potential between phases. For vapour-liquid equilibrium this could be written as

$$\begin{align}
T^\mathrm{vap} &= T^\mathrm{liq}\tag{1}\\
p^\mathrm{vap} &= p^\mathrm{liq}\tag{2}\\
\mu^\mathrm{vap} &= \mu^\mathrm{liq}~.\tag{3}
\end{align}$$

To distinguish between the phases, we must use a property that takes a different value in each. A suitable choice is the specific volume, or density. This provides two "iteration" variables: $v^\mathrm{liq}$ and $v^\mathrm{vap}$.

As we are iterating using two variables, we need to select two of the three equations to "complete" our problem. While any two could be used, solving for temperature is far more expensive than solving for pressure or chemical potential. This is because equations of state are generally not analytically solvable for temperature, meaning they would require an additional inner loop to solve for this. Therefore we solve for equilibrium using equations (2) and (3).

This is a root-finding problem in $\mathbb R^2$, with an objective function $F$ defined as

$$F : \mathbb R^{2} \to \mathbb R^{2}$$

$$F : \left[\begin{split}
v^\mathrm{liq}\\
v^\mathrm{vap}
\end{split}\right]
\to
\left[\begin{split}
p^\mathrm{liq} - p^\mathrm{vap} = 0\\
\mu^\mathrm{liq} - \mu^\mathrm{vap} = 0
\end{split}\right]~.$$
"""

# ╔═╡ 2f60da7d-a5aa-424e-921b-dc6377423d50
md"""
## Scaling factors
When solving our problem, properly scaling our equations is very important. Part of this is due to [conditioning](https://en.wikipedia.org/wiki/Condition_number) [^1], and part due to how we define convergence. In a root-finding problem, convergence is usually declared when the magnitude of the function falls below a specified value, written as

$$|f(x)| < ϵ$$

where ϵ is a **user-defined tolerance**. By using scaling factors, our variables are similarly-sized, and the system of equations is better behaved. The typical scaling factors for the thermodynamic variables are given by the table below.
"""

# ╔═╡ d6f55d1e-a9c4-49c5-9ec3-14d51d0781e8
@htl("""
		<table>
		<caption>Typical scaling factors for thermodynamic variables</caption>
		  <tr>
		    <td></td>
		    <th>Cubic</th>
		    <th>SAFT</th>
		  </tr>
		  <tr>
		    <th>Pressure</th>
		    <td><i>P</i><sub>c</sub></td>
		    <td><i>R⋅ϵ</i>/(<i>N</i><sub>A</sub>⋅<i>σ</i>³)</td>
		  </tr>
		  <tr>
		    <th>Temperature</th>
		    <td><i>T</i><sub>c</sub></td>
		    <td><i>ϵ</i></td>
		  </tr>
		  <tr>
			<th>Molar Energies (e.g. <i>μ</i>, <i>g</i>, <i>a</i>)</th>
		    <td><i>R⋅T</i></td>
			<td><i>R⋅T</i></td>
		  </tr>
		</table>
		""")

# ╔═╡ 103b01a4-971a-44b1-be08-b3e59d952dcb
md"""
Using Clapeyron, we can access the pressure and temperature scaling factors using 

```julia
ps = p_scale(model)
Ts = T_scale(model)
```
"""

# ╔═╡ b7b778eb-6f77-4183-8c67-76cb2aa3c25c
md"""
Another technique often adopted is using logs to scale variables scanning multiple orders of magnitude. This frequently arises in relation to volumes, with liquid and vapour volumes varying by up to 1000x.
"""

# ╔═╡ 61c34a9b-1e5f-45ab-bbfd-c4e8b21c7320
md"""

## Initial guesses

As with all numerical methods, good initial guesses are quite important. To obtain these, we can leverage the theory of corresponding states. This can be stated as: "All fluids, when compared at the same reduced temperature and reduced pressure, have approximately the same compressibility factor and all deviate from ideal gas behavior to about the same degree." [^1]. This allows us to express the solution to the van der Waals EoS saturation curve in terms of **reduced variables**. We use a highly-accurate numerical approximation for this, with saturated volumes given by

$$\begin{gather}
v^\textrm{sat. liq vdW} = 3b/c^\textrm{sat. liq vdW}\\
v^\textrm{sat. vap vdW} = 3b/c^\textrm{sat. vap vdW}
\end{gather}$$

where $c$ is given by 

$$c^\textrm{sat. liq vdW} = 1 + 2(1-T_\mathrm{r})^{1/2} + \frac25(1-T_\mathrm{r}) - \frac{13}{25}(1-T_\mathrm{r})^{3/2} + 0.115(1-T_\mathrm{r})^2$$

$$c^\textrm{sat. vap vdW} = \begin{cases} 2(1+\frac25(1-T_\mathrm{r}) + 0.161(1-T_\mathrm{r})^2) - c^\textrm{sat. liq}\quad 0.25 < T_\mathrm{r} \leq 1\\ 2(\frac32 - \frac49T_\mathrm{r}-0.15T_\mathrm{r}^2)-c^\textrm{sat. liq}\quad0\leq T_\mathrm{r} < 0.64\end{cases}$$

A potential issue when relying on the van der Waals saturation curve to determine our initial guesses is the situation where the initial guesses fall _inside_ the saturation envelope of the fluid we're solving for, as that can lead to numerical instability and convergence issues. By choosing guesses that bracket, but don't lie too far from, the saturation curve, convergence is far more reliable.

To do so, I chose to use an extra factor of 0.5 for the liquid volume and 2 for the vapour volume:

$$\begin{gather}
v^\textrm{sat. liq}_0 = 0.5v^\textrm{sat. liq vdW}\\
v^\textrm{sat. vap}_0 = 2v^\textrm{sat. vap vdW}
\end{gather}$$

 This is done to avoid the guesses falling _inside_ the saturation curve of the fluid we're solving for, as that can lead to numerical instability and convergence issues. 

The derivation of the volume expression can be seen in the article [_The van der Waals equation: analytical and approximate
solutions_](https://rdcu.be/cTmlc). The code is implemented below:
"""

# ╔═╡ 80662d56-e97c-4ea5-9ba7-0d2ad827740d
md"""
Because we only have temperature specified, we cannot use the ideal gas equation without requiring another correlation to generate a guess for the saturation pressure. To avoid this we estimate a vapour volume of 

$$V^\mathrm{vap}_0 = -2B(T)$$

where $B$ is the second virial coefficient.

For liquids our approach differs depending on the class of model we're using. For cubic equations of state the $b$ parameter can be used [^2]

$$V^\mathrm{liq}_0 = b$$ 

where $b$ corresponds to the $b$ parameter within the repulsive term of the equation of state. For SAFT and its derivatives, we can use the same approach we used in Section 3.1, where the initial guess is defined as

$$V_0^\mathrm{liq} = \frac{\pi}{6}\cdot N_A \cdot \sigma^3~.$$
""";

# ╔═╡ 07d75f58-ddd5-436a-a457-de61e571dac6
md"""

## Implementation
Now, lets implement our saturation solver. We have two functions -- our objective function, or the function to be zeroed, and the solver.
"""

# ╔═╡ 94e20664-95f2-4760-9bc4-b81df13328dc
"""
	sat_p_objective(model, T, V)

Defines the objective function for a pure saturation solver. Returns a vector ```[f1, f2]```, where ```f1``` is the pressure equation, and ```f2``` is the chemical potential equation.
"""
function sat_p_objective(model, T, V)
	# Unpack input array
	Vl, Vv = V

	# Define in-line equations for pressure and chemical potential
	p(V) = pressure(model, V, T)
	μ(V) = VT_chemical_potential(model, V, T)

	# Calculate the objective function
	f1 = (p(Vl) - p(Vv))/p_scale(model)
	f2 = (μ(Vl) - μ(Vv))/(R*T)
	return [f1, f2[1]]
end

# ╔═╡ 822b518b-8ebf-4bf5-b236-6282ce6a0c27
md"""
Now, we just need to define a model and test out our solver!
"""

# ╔═╡ 584406e0-f41c-4043-b207-ed5a39ef9d88
md"""
And we can compare it to the Clapeyron result
"""

# ╔═╡ 2ffa15ba-9c9c-4d9e-b154-08bb251d0749
md"""
## Building phase diagrams

Now we are able to determine the location of the saturation curve, how do we build up a graph of the entire phase boundary? We can call the solver we just wrote above again and again for different temperatures with the same initial guess, and for most cases it will probably converge. However, as before there are a particular number of difficulties near the critical point and the solver can become very sensitive to the initial guesses. 

To help with this, we can reuse each previous result as the new guess to the solver. On top of being very important near the critical point, this technique is very important for speeding up the overall solver -- the very good initial guesses obtained in this way allow for rapid convergence.

When building this, remember that the triple point is not represented by typical equations of state! As only liquid and vapour phases are captured, the only significant point we see on pure phase diagrams is the critical point.
"""

# ╔═╡ 65a528b6-eebb-422c-9221-f2bd9df0d2d2
md"""
## Critical point solver

### Formulation

Above, we traced the pure saturation envelope between two user-decided points. If one of these falls above the critical point, the saturation solver will either fail or converge to a trivial solution [^3]. If we want to determine the end point of our saturation curve before beginning calculation, how should we go about this?

For a cubic equation of state the critical temperature and pressure are input to the equation of state, meaning for a pure substance you will always know the critical point beforehand. As this is not the case for other equations of state, we must solve numerically for $p_\mathrm{c}$ and $T_\mathrm{c}$.

When considering the projection into _p,v_ space, the critical point of a pure substance can be defined as the point of inflection along a unique isotherm (known as the _critical isotherm_). Therefore, we can write three equations defining this point exactly:

$$\begin{gather}
f_1(v, T) = \left(\frac{\partial p(v,T)}{\partial v}\right)_{T} = 0\tag{4}\\
f_2(v, T) = \left(\frac{\partial^2 p(v,T)}{\partial v^2}\right)_{T} = 0\tag{5}\\
f_3(v, T) = \left(\frac{\partial^3 p(v,T)}{\partial v^3}\right)_{T} < 0\tag{6}\\
\end{gather}$$

however, as we iterate using two variables (the specific volumes of each phase), we select only equations (4) and (5) to complete our root-finding problem. (To be strict, once the solution has been found, equation (6) should also be checked for consistency. In practice, however, this is not normally done.)

To define our objective function, we would usually require analytical derivatives, as calculation with finite differences [^4] are both inaccurate and expensive, especially for derivatives beyond second order. However, automatic differentiation allows us to easily determine the exact derivatives of the objective function, as well as the higher-order derivatives needed to solve this using Newton's method. Note that although numerical methods that do not rely on derivatives exist, it is faster to use derivative information if it is available.
"""

# ╔═╡ 1e01206f-e565-4d90-83f8-f7c038b5e961
md"""
### Initial guesses
The final issue we must resolve is the generation of **initial guesses**. Luckily, a critical point solver isn't too sensitive to initial guesses.

The volume guess is an empirically chosen packing fraction of $0.3$. This corresponds to 

$$V_0^\mathrm{liq} = \frac{1}{0.3}\cdot\frac{\pi}{6}\cdot N_A \cdot \sigma^3$$

as defined in Section 3.1.

The temperature guess corresponds to

$$T_0 = 2\cdot T^\mathrm{scale}$$

where for SAFT,

$T^\mathrm{scale} = ϵ~.$
"""

# ╔═╡ ea2d00b5-0d10-4109-aec1-631a4575fc06
md"""
### Implementation

Rather than write the entire solver by hand, we will now rely on Newton's method exported by the NLsolve library. The format of this function is:

```julia
nlsolve(f!, initial_x, autodiff = :forward, method = :newton)
```

Note that ``!`` signifies an [in-place function](https://en.wikipedia.org/wiki/In-place_algorithm). This means that rather than returning a value, it updates the first value passed to the new values.
"""

# ╔═╡ c53316bd-ecbd-43b0-b016-99a1985fc6e3
"""
	critical_objective!(model, F, x)

Defines the objective function for a critical point solver. Returns a vector ```[f1, f2]```, where ```f1``` is the first pressure derivative, and ```f2``` is the second pressure derivative.
"""
function critical_objective!(model, F, x)
	V = x[1]
	T = x[2]
	p(V) = pressure(model, V, T)/p_scale(model)
	f1(V) = ForwardDiff.derivative(p, V)
	f2(V) = ForwardDiff.derivative(f1, V)
	F .= [f1(V), f2(V)]
end

# ╔═╡ 7ef032f6-a68f-43d5-9152-bcf64c336e80
"""
	critical_point_guess(model::SAFTModel)

Generates initial guesses for the critical point of a fluid. Scales the limit of the packing fraction and temperature scale factor using empirical factors.
"""
function critical_point_guess(model::SAFTModel)
	σ = model.params.sigma.diagvalues[1]
	seg = model.params.segment.values[1]
	# η = 0.3
	V0 = 1/0.3 * 1.25 * π/6 * N_A * σ^3 * seg
	return [log10(V0), 2.0]
end

# ╔═╡ 6dbba706-0e29-4e9b-8525-2d581706499d
"""
	solve_critical_point(model)

Directly solves for the critical point of an equation of state using automatic differentiation and Newton's method.
"""
function solve_critical_point(model)
	Ts = T_scale(model)
	# Generate initial guesses
	x0 = critical_point_guess(model)
	
	# Solve system for critical point
	res = nlsolve((F, x) -> critical_objective!(model, F, [exp10(x[1]), Ts*x[2]]), x0, autodiff = :forward, xtol=1e-9, method=:newton)	

	# Extract answer
	Vc = exp10(res.zero[1])
	Tc = Ts*res.zero[2]
	# Calculate pressure
	pc = pressure(model, Vc, Tc)
	# Return values
	return (pc, Tc, Vc)
end

# ╔═╡ a59304ac-2480-4529-97aa-80d472e540cf
md"""
We can now use our function to calculate the critical point
"""

# ╔═╡ 5f5b5e95-f47e-4dc4-97cc-53b774df3638
begin
	model_SAFT = PCSAFT(["water"])
	pc, Tc, Vc = solve_critical_point(model_SAFT)
end

# ╔═╡ a9eb545c-10dd-4e4e-ab41-f6e14ffdcede
begin
	try
		@htl("""
		<table>
		<caption>Solver Results</caption>
		  <tr>
		    <td>Critical Temperature</td>
		    <td>$(round(Tc; sigdigits=5))</td>
		    <td>K</td>
		  </tr>
		  <tr>
		    <td>Critical Volume</td>
		    <td>$(round(Vc; sigdigits=4))</td>
		    <td>m³</td>
		  </tr>
		  <tr>
		    <td>Critical Pressure</td>
		    <td>$(round(pc/1e6; sigdigits=4))</td>
		    <td>MPa</td>
		  </tr>
		</table>
		""")
	catch 
		@htl("""
		<table>
		<caption>Solver Results</caption>
		  <tr>
		    <td>Critical Temperature</td>
		    <td>?</td>
		    <td>K</td>
		  </tr>
		  <tr>
		    <td>Critical Volume</td>
		    <td>?</td>
		    <td>Pa</td>
		  </tr>
		</table>
		""")
	end
end

# ╔═╡ f3328063-1529-41b8-99b1-372f157316d2
md"""
As with the volume solver, we can compare this to the implementation within Clapeyron using the ```crit_pure``` function.

```
(Tc, pc, Vc) = crit_pure(model)
```
"""

# ╔═╡ ecd34ead-d239-40a2-a997-d3eaea6b69cd
Tc_Clapeyron, pc_Clapeyron, Vc_Clapeyron = crit_pure(model_SAFT)

# ╔═╡ 75681c1d-b294-406d-8dcd-138533ea0205
Tc ≈ Tc_Clapeyron

# ╔═╡ 8b306d79-33a4-4b76-ad74-e60ffe9eca6f
pc ≈ pc_Clapeyron

# ╔═╡ 90fd5f17-4347-416c-95a5-d6791ccec7b8
Vc ≈ Vc_Clapeyron

# ╔═╡ f87fb61e-2f63-4df8-8d31-e397966f840f
md"""
And we've converged correctly on the right answer!
"""

# ╔═╡ 58b76139-6976-4624-8d71-347b50e1b494
md"""
# Footnotes
[^1]: Equation conditioning generally refers to the sensitivity of the output to small pertubations in the input. Often in numerical methods, a highly sensitive (i.e. poorly conditioned) problem can result in the loss of precision. The condition number is often discussed for both matrices and for functions.

[^2]: If you are using a model with volume translation, you should also account for that in your initial guess.

[^3]: A trivial solution is encountered when the solver converges both phases to the same solution, resulting in the automatic satisfaction of the equilibrium conditions (equality of temperature, pressure, and chemical potential)

[^4]: Finite differencing is the traditional method for calculation of derivatives. A good explanation can be found on [wikipedia](https://en.wikipedia.org/wiki/Finite_difference).
"""

# ╔═╡ 1d24ec62-7457-4b75-b2b6-740711df3e49
#The essence of automatic differentiation is tracing the elementary operations, like addition and subtraction, that happen to a given variable through some code. This can then be differentiated directly and combined using the chain rule. There are two main "modes" of automatic differentiation. Forward-mode and Reverse-mode. To learn more about these, I recommend taking a look at the [wikipedia page](https://en.wikipedia.org/wiki/Automatic_differentiation) as well as the relevant notes from the course 18.337 at MIT; [forward-mode notes](https://book.sciml.ai/notes/08/) and [reverse-mode notes](https://book.sciml.ai/notes/10/).

# ╔═╡ 33ed3f41-3107-4497-85e0-aa7de6686612
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]));

# ╔═╡ bb419121-16b9-4344-b245-3e19f8d4830a
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]));

# ╔═╡ bb4b3b1b-702b-4462-967d-2d25bfe3f226
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]));

# ╔═╡ 7c415b6a-cb43-4d7e-8a5f-76a0e510cabd
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));

# ╔═╡ f7066d86-00f8-441f-868b-7c94037b36ac
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."];

# ╔═╡ 02759a14-8161-4332-b960-3d3a5f052d19
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]));

# ╔═╡ 704cb182-fa73-4cd8-bad4-16118a8f8219
if Tc ≈ Tc_Clapeyron
	correct()
else
	almost(md"Not quite there, you're off by δ = $(Tc - Tc_Clapeyron)")
end

# ╔═╡ a676acf4-e167-425c-b63e-9c9fb8c24d13
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define **$(Markdown.Code(string(variable_name)))**"]));

# ╔═╡ 2755df96-8731-440d-8b5e-4e4dd787f55c
vdW_b(pc, Tc) = 1/8 * (R*Tc)/pc;

# ╔═╡ fe3ecf69-f3db-492f-a0ae-bc0622256071
"""
	vdW_saturation_volume(model, T)

Returns the saturation curve for a van der Waals with an equivalent critical temperature and pressure for ```model```. Uses a derivation from 10.1007/s10910-007-9272-4 for the approximate solution to the saturation curve.
"""
function vdW_saturation_volume(model, T)
	Tc, pc, vc = crit_pure(model)
	Tr = T/Tc
	
	# Valid for 0 ≤ Tᵣ ≤ 1
	cL = 1 + 2(1-Tr)^(1/2) + 2/5*(1-Tr) - 13/25*(1-Tr)^(3/2) + 0.115*(1-Tr)^2

	if 0.25 < Tr ≤ 1
		# Valid for 0.25 < Tr ≤ 1
		cG = 2*(1 + 2/5*(1-Tr) + 0.161*(1-Tr)^2) - cL
	elseif 0 ≤ Tr < 0.64
		# Valid for 0 ≤ Tr < 0.64
		cG = 2*(3/2 - 4/9*Tr - 0.15Tr^2) - cL
	else
		@error "Invalid reduced temperature, Tr = $Tr. Use a value between 0 and 1"
	end
	b = vdW_b(pc, Tc)
	vL = 3b/cL
	vG = 3b/cG
	return [0.5*vL, 2*vG]
end

# ╔═╡ fe3c0403-5447-4171-a536-5996cc11c77d
"""
	solve_sat_p(model, T; V0 = vdW_saturation_volume(model, T), itersmax=100, abstol=1e-10)

Solves an equation of state for the saturation pressure at a given temperature using Newton's method. By default uses the solution to the van der Waals equation for initial guesses. Returns (psat, _v_\\_liq, _v_\\_vap)
"""
function solve_sat_p(model, T; V0 = vdW_saturation_volume(model, T), itersmax=100, abstol=1e-10)
	# Objective function accepting a vector of volumes, R²→R² 
	f(logV) = sat_p_objective(model, T, exp10.(logV))
	# function returning the Jacobian of our solution, R²→R²ˣ²
	Jf(logV) = ForwardDiff.jacobian(f, logV)

	logV0 = log10.(V0)
	logVold = 0.0
	logV = logV0
	fx = 1.0
	fx0 = f(logV0)
	iters = 0
	
	# Iterate until converged or the loop has reached the maximum number of iterations
	while (iters < itersmax && all(abs.(fx) .> abstol))
		Jfx = Jf(logV) # Calculate the jacobian
		fx = f(logV) # Calculate the value of f at V
		d = -Jfx\fx # Calculate the newton step
		logVold = logV # Store current iteration
		logV = logV .+ d # Take newton step
		iters += 1 # Increment our iteration counter
	end
	
	# Show a warning if the solver did not converge (uses short circuit evaluation rather than if statement)
	iters == itersmax && @warn "solver did not converge in $(iters) iterations\nfV=$(fx)"
	
	V = exp10.(logV)
	p_sat = pressure(model, V[1], T)
	return (p_sat, V[1], V[2])
end

# ╔═╡ 03a0b678-0ad6-4c5a-b1d8-6497c9703ccb
begin
	# Specify our state
	cubic_model = PR(["hexane"])
	T = 373.15 # K
	# Solve the nonlinear system 
	(p_sat, V_liq, V_vap) = solve_sat_p(cubic_model, T)
end

# ╔═╡ 5f81ba8b-b7db-4571-b797-1a4ea06fa9a7
begin
	try
		@htl("""
		<table>
		<caption>Solver Results</caption>
		  <tr>
		    <td>Temperature</td>
		    <td>$(round(T; sigdigits=5))</td>
		    <td>K</td>
		  </tr>
		  <tr>
		    <td>Saturation pressure</td>
		    <td>$(round(p_sat; sigdigits=4))</td>
		    <td>Pa</td>
		  </tr>
		  <tr>
			<td>Liquid volume</td>
		    <td>$(@sprintf "%.2e"  V_liq)</td>
			<td>m³</td>
		  </tr>
		  <tr>
			<td>Vapour volume</td>
		    <td>$(@sprintf "%.2e" V_vap)</td>
			<td>m³</td>
		  </tr>
		</table>
		""")
	catch
		@htl("""
		<table>
		<caption>Solver Results</caption>
		  <tr>
		    <td>Temperature</td>
		    <td>?</td>
		    <td>K</td>
		  </tr>
		  <tr>
		    <td>Saturation pressure</td>
		    <td>?</td>
		    <td>Pa</td>
		  </tr>
		  <tr>
			<td>Liquid volume</td>
		    <td>?</td>
			<td>m³</td>
		  </tr>
		  <tr>
			<td>Vapour volume</td>
		    <td>?</td>
			<td>m³</td>
		  </tr>
		</table>
		""")
	end
end

# ╔═╡ 61207fa8-b9cb-43dc-9130-1a8bbf7cf640
psat_Clapeyron, Vlsat_Clapeyron, Vvsat_Clapeyron = saturation_pressure(cubic_model, T)

# ╔═╡ 00722d62-afed-4dbb-951d-9de89cba8df0
p_sat ≈ psat_Clapeyron

# ╔═╡ 78d8727b-e04d-4244-8e0a-c650cf3d11cd
V_liq ≈ Vlsat_Clapeyron

# ╔═╡ 9038ceda-a8b3-45ee-bdc0-e9052563f727
V_vap ≈ Vvsat_Clapeyron

# ╔═╡ 59927acf-9a17-4117-870c-5cc19169311d
let
	Tcrit, pcrit, vcrit = crit_pure(cubic_model)
	Ts = LinRange(280.0, 1.5*Tcrit, 500)
	Ts2 = Ts[Ts .> 400]
	ps = LinRange(1e5, 1.5*pcrit, 500)

	satp = zeros(length(Ts2))
	Vlsat = zeros(length(Ts2))
	Vvsat = zeros(length(Ts2))
	for (i, T) in enumerate(Ts2)
		(satp[i], Vlsat[i], Vvsat[i]) = saturation_pressure(cubic_model, T)
	end
	
	plotlyjs()
	
	surface(ps./1e6, Ts, (x, y) -> log10(volume(cubic_model, 1e6x, y)), c=:summer, xlabel="p / MPa", ylabel="T / K", zlabel="log10(v / m³/mol)", camera=(45, 90), colorbar=false)
	scatter!([pcrit/1e6], [Tcrit], log10.([vcrit]), label="critical point", color=2)
	plot!(repeat(satp./1e6, 2), repeat(Ts2, 2), log10.(vcat(Vlsat, Vvsat)), width=5, color=1, label="saturation envelope")
end

# ╔═╡ bdd0973b-2491-4ad7-a6e0-3bec21e206cd
if all(solve_sat_p(cubic_model, T) .≈ saturation_pressure(cubic_model, T))
	correct()
else
	almost(md"Not quite there, you're off by δ = $(p_sat - psat_Clapeyron)")
end

# ╔═╡ c5109b81-887a-46e6-92da-10bda0397380
begin
	# The vector of temperature values
	crit_temp, _, _ = crit_pure(cubic_model)
	T_vec = range(285.0, 0.9999crit_temp, length=1000)
	# Preallocate our pressure and volume vectors
	p_vec = zeros(length(T_vec))
	Vl_vec = zeros(length(T_vec))
	Vv_vec = zeros(length(T_vec))

	# Create initial guess
	V0 = vdW_saturation_volume(cubic_model, T)
	for (i, T) in enumerate(T_vec)
		try
			sat = solve_sat_p(cubic_model, T; V0=V0)
			
			if ~any(isnan.(sat))
				(p_vec[i], Vl_vec[i], Vv_vec[i]) = sat
				V0 = [Vl_vec[i], Vv_vec[i]] # Store previous iteration for new guess
			end
		catch
			continue
		end
	end
	T_vec = T_vec[.!iszero.(p_vec)]
	filter!.(!iszero, [p_vec, Vl_vec, Vv_vec])
end

# ╔═╡ 1f218041-692c-4d1c-b873-39bdf7c45ccd
let
	Tc, pc, Vc = crit_pure(cubic_model)
	gr()
	plot(title="PT plot for water", xlabel="T (K)", ylabel="P (MPa)", framestyle=:box, tick_direction=:out, grid=:off, legend=:topleft)

	plot!(T_vec, p_vec./1e6, label="Saturation curve", linewidth=2)
	scatter!([Tc], [pc]./1e6, label="Critical point", markersize=4.5, color=2)
end

# ╔═╡ 4981857e-e33a-43fa-b4f3-d32db350a4e2
let
	Tc, pc, Vc = crit_pure(cubic_model)
	gr()
	plot(xaxis=:log, title="VT plot for water", xlabel="V (m³)", ylabel="T (K)", framestyle=:box, tick_direction=:out, grid=:off)

	plot!(Vl_vec, T_vec, label="", linewidth=2, color=1)
	plot!(Vv_vec, T_vec, label="Saturation envelope", linewidth=2, color=1)
	scatter!([Vc], [Tc], label="Critical point", markersize=4.5, color=2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clapeyron = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"

[compat]
Clapeyron = "~0.3.7"
ForwardDiff = "~0.10.32"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
NLsolve = "~4.5.1"
PlotlyJS = "~0.18.8"
Plots = "~1.31.6"
PlutoUI = "~0.7.39"
Roots = "~2.0.2"
ShortCodes = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "802688e3c27045e63434127f015fb7866a9eebc5"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.BlackBoxOptim]]
deps = ["CPUTime", "Compat", "Distributed", "Distributions", "HTTP", "JSON", "LinearAlgebra", "Printf", "Random", "SpatialIndexing", "StatsBase"]
git-tree-sha1 = "41e347c63757dde7d22b2665b4efe835571983d4"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.1"

[[deps.Blink]]
deps = ["Base64", "BinDeps", "Distributed", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Reexport", "Sockets", "WebIO", "WebSockets"]
git-tree-sha1 = "08d0b679fd7caa49e2bca9214b131289e19808c0"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.5"

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
version = "0.5.2+0"

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
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

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
git-tree-sha1 = "5a2cff9b6b77b33b89f3d97a4d367747adce647e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.15.0"

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
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

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

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

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

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

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

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

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

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "fd6f0cae36f42525567108a42c1c674af2ac620d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

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

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "1e566ae913a57d0062ff1af54d2697b9344b99cd"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.14"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "Pkg", "Sockets", "WebSockets"]
git-tree-sha1 = "82dfb2cead9895e10ee1b0ca37a01088456c4364"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "0.7.6"

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
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

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

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "180d744848ba316a3d0fdf4dbd34b77c7242963a"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.18"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "53d6325e14d3bdb85fd387a085075f36082f35a3"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.8"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "79830c17fe30f234931767238c584b3a75b3329d"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.6"

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
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

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
version = "0.7.0"

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
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e5364b687e552d73543cf09e583b944eaffff6c4"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShortCodes]]
deps = ["Base64", "CodecZlib", "HTTP", "JSON3", "Memoize", "UUIDs"]
git-tree-sha1 = "0fcc38215160e0a964e9b0f0c25dcca3b2112ad1"
uuid = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
version = "0.3.3"

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
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

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
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

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

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

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

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "a8bbcd0b08061bba794c56fb78426e96e114ae7f"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.18"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

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
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

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
version = "5.1.1+0"

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
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─58b0aaa8-f47a-11ec-1113-9933151340be
# ╟─c4b5bc07-7f17-47d2-a3e6-945813e29b35
# ╟─2f60da7d-a5aa-424e-921b-dc6377423d50
# ╟─d6f55d1e-a9c4-49c5-9ec3-14d51d0781e8
# ╟─103b01a4-971a-44b1-be08-b3e59d952dcb
# ╟─b7b778eb-6f77-4183-8c67-76cb2aa3c25c
# ╟─61c34a9b-1e5f-45ab-bbfd-c4e8b21c7320
# ╟─80662d56-e97c-4ea5-9ba7-0d2ad827740d
# ╠═fe3ecf69-f3db-492f-a0ae-bc0622256071
# ╟─07d75f58-ddd5-436a-a457-de61e571dac6
# ╠═94e20664-95f2-4760-9bc4-b81df13328dc
# ╠═fe3c0403-5447-4171-a536-5996cc11c77d
# ╟─822b518b-8ebf-4bf5-b236-6282ce6a0c27
# ╠═03a0b678-0ad6-4c5a-b1d8-6497c9703ccb
# ╟─5f81ba8b-b7db-4571-b797-1a4ea06fa9a7
# ╟─584406e0-f41c-4043-b207-ed5a39ef9d88
# ╠═61207fa8-b9cb-43dc-9130-1a8bbf7cf640
# ╠═00722d62-afed-4dbb-951d-9de89cba8df0
# ╠═78d8727b-e04d-4244-8e0a-c650cf3d11cd
# ╠═9038ceda-a8b3-45ee-bdc0-e9052563f727
# ╟─bdd0973b-2491-4ad7-a6e0-3bec21e206cd
# ╟─2ffa15ba-9c9c-4d9e-b154-08bb251d0749
# ╠═c5109b81-887a-46e6-92da-10bda0397380
# ╟─1f218041-692c-4d1c-b873-39bdf7c45ccd
# ╟─4981857e-e33a-43fa-b4f3-d32db350a4e2
# ╟─59927acf-9a17-4117-870c-5cc19169311d
# ╟─65a528b6-eebb-422c-9221-f2bd9df0d2d2
# ╟─1e01206f-e565-4d90-83f8-f7c038b5e961
# ╟─ea2d00b5-0d10-4109-aec1-631a4575fc06
# ╠═c53316bd-ecbd-43b0-b016-99a1985fc6e3
# ╠═7ef032f6-a68f-43d5-9152-bcf64c336e80
# ╠═6dbba706-0e29-4e9b-8525-2d581706499d
# ╟─a59304ac-2480-4529-97aa-80d472e540cf
# ╠═5f5b5e95-f47e-4dc4-97cc-53b774df3638
# ╟─a9eb545c-10dd-4e4e-ab41-f6e14ffdcede
# ╟─f3328063-1529-41b8-99b1-372f157316d2
# ╠═ecd34ead-d239-40a2-a997-d3eaea6b69cd
# ╠═75681c1d-b294-406d-8dcd-138533ea0205
# ╠═8b306d79-33a4-4b76-ad74-e60ffe9eca6f
# ╠═90fd5f17-4347-416c-95a5-d6791ccec7b8
# ╟─704cb182-fa73-4cd8-bad4-16118a8f8219
# ╟─f87fb61e-2f63-4df8-8d31-e397966f840f
# ╟─58b76139-6976-4624-8d71-347b50e1b494
# ╟─1d24ec62-7457-4b75-b2b6-740711df3e49
# ╟─33ed3f41-3107-4497-85e0-aa7de6686612
# ╟─bb419121-16b9-4344-b245-3e19f8d4830a
# ╟─bb4b3b1b-702b-4462-967d-2d25bfe3f226
# ╟─7c415b6a-cb43-4d7e-8a5f-76a0e510cabd
# ╟─f7066d86-00f8-441f-868b-7c94037b36ac
# ╟─02759a14-8161-4332-b960-3d3a5f052d19
# ╟─a676acf4-e167-425c-b63e-9c9fb8c24d13
# ╟─2755df96-8731-440d-8b5e-4e4dd787f55c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
