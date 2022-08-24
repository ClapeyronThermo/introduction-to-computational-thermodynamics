### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 23962934-2638-4788-9677-ae42245801ec
begin
	using Clapeyron: VT_isothermal_compressibility, VT_chemical_potential, R̄, ABCubicModel, SAFTModel, N_A 
	const R = R̄
	using Clapeyron, ForwardDiff, Roots, Optim, LinearAlgebra, PolynomialRoots # Numerical packages
	using LaTeXStrings, Plots, ShortCodes, Printf # Display and plotting
	using HypertextLiteral
	using PlutoUI
	using Unitful
	PlutoUI.TableOfContents()
end

# ╔═╡ 30f48408-f16e-11ec-3d6b-650f1bf7f435
md"""
# Section 3.1 - Volume Solvers

Now we know how to choose an equation of state for different situations, we need to investigate how to obtain the fluid properties at a specified state. To start with, we will look at how to solve for the volume at a given pressure, temperature, and phase for a pure fluid. From there, we will see how this changes when the phase is not known beforehand, and when dealing with a multicomponent mixture.

Usually, we specify pressure and temperature (_pT_), which when solving for the volume corresponds to solving the equation

$$p(v,T_0) = p_0$$

for all values of $v$. These are considered the **volume roots** for our equation of state.

## Cubic EoS

As you have seen in section 2, the general cubic equation of state can be written as

$$p = \frac{RT}{v-b} - \frac{a\alpha(T)}{(v + \delta_1 b)(v + \delta_2b)}$$

where the values of $\delta_1, \delta_2$ are specific to each equation of state (van der Waals, Peng Robinson, Redlich Kwong), and the form of the **alpha function**, $a(T)$, can be selected to match specific types of components or problems. In this workbook we will consider the van der Waals EoS, which is the case when 

$$\delta_1 = \delta_2 = 0$$
$$\alpha(T) = 1$$
"""

# ╔═╡ 2f3a7b9d-3dac-4e12-abb1-ce09cef36e93
md"""
Giving us

$$p = \frac{RT}{v-b} - \frac{a}{v^2}$$

where

$$\begin{align} 
a &= \frac{27}{64}\frac{(RT_\mathrm{c})^2}{p_\mathrm{c}}\\
b &= \frac{1}{8}\frac{RT_\mathrm{c}}{p_\mathrm{c}}\\
\end{align}$$

To solve this for the volume, we can leverage the fact you can rearrange it as a cubic equation in terms of the **compressibility factor**, Z.

To do this, we multiply the expression through by the denominators and substitute in the definition of Z.

$$Z = \frac{Pv}{RT}$$

$$\begin{gather}
pv^2(v-b) = RTv^2 - a(v-b)\\
pv^3 - (bp + RT)v^2 + av - ab = 0\\
Z^3 - \left(1 + \frac{bp}{RT}\right)Z^2 + \left(\frac{ap}{(RT)^2}\right)Z - \frac{abp^2}{(RT)^3} = 0
\end{gather}$$

To express this cubic neatly we define 2 constants, $A$ and $B$.

$$\begin{align} 
A &= a\cdot\frac{p}{(RT)^2}\\
B &= b\cdot\frac{p}{RT}
\end{align}$$

The cubic we then need to solve is

$$Z^3 - (1 + B)Z^2 + AZ - AB = 0$$

A similar result can be obtained for any cubic equation.
"""

# ╔═╡ 663f44e1-6f7c-490c-9f7a-446f94a63c2a
md"""
The three roots for the van der Waals equation of state can be seen below:
"""

# ╔═╡ 2807debe-5c79-40a7-acf3-947b288449d7
md"""
T = 
"""

# ╔═╡ 591b6a0b-548b-44f4-8415-16b8aa219c0b
@bind T_plt1 PlutoUI.Slider(250u"K":1u"K":325u"K", show_value=true, default=275u"K")

# ╔═╡ fee14db9-d42d-4a6b-8ef3-a539421be776
let
	function cubic_volume(model, p, T)
		Tc, pc, _ = crit_pure(model)
	
		a = 27/64 * (R*Tc)^2/pc
		b = 1/8 * (R*Tc)/pc
		
		A = a*p/(R*T)^2
		B = b*p/(R*T)
		
		poly = [-A*B, A, -(1+B), 1.0]
		Zvec = roots(poly)
		Vvec = Zvec.*(R*T/p)
		return Vvec
	end
	
	model = vdW(["carbon dioxide"])
	R = 8.314
	
	a, b = model.params.a.values[1], model.params.b.values[1]
	# p_raw(v, T) = R*T/(v-b) - a/v^2
	p_raw(v, T) = pressure(model, v, T)
	
	Tcrit, pcrit, vcrit = crit_pure(model)
	Tsat = LinRange(200.0, 0.9999Tcrit, 500)
	# ps = LinRange(1e5, 0.9999pcrit, 500)

	psat = zeros(length(Tsat))
	Vlsat = zeros(length(Tsat))
	Vvsat = zeros(length(Tsat))
	for (i, T) in enumerate(Tsat)
		(psat[i], Vlsat[i], Vvsat[i]) = saturation_pressure(model, T)
	end

	# Z = pV/RT
	Zlsat = @. psat*Vlsat/(R*Tsat)
	Zvsat = @. psat*Vvsat/(R*Tsat)

	psat = psat./1e6
	pcrit = pcrit/1e6
	gr()
	plt = plot(
		title="pv diagram for CO₂ using the van der Waals EoS",
		ylabel="p (MPa)", xlabel="v (m³/mol)",
		framestyle=:box, tick_direction=:out, grid=:off,
		xscale=:log,
		ylim = (minimum(psat), 1.1pcrit)
	)
	# plot!(vcat(Vvsat, Vlsat), vcat(psat, psat)./1e6)
	plot!(
		Vlsat, psat,
		linewidth=2, color=1, linestyle=:solid,
		label="Saturation curve"
	)
	plot!(
		Vvsat, psat,
		linewidth=2, color=1, linestyle=:solid,
		label=""
	)
	scatter!([vcrit], [pcrit], label="critical point", markersize=5.5, color=3)

	T1 = ustrip(T_plt1)
	psat1, Vlsat1, Vvsat1 = saturation_pressure(model, T1)
	
	plot!(
		v -> p_raw(v, T1)/1e6,
		range(minimum(Vlsat), maximum(Vvsat), 500),
		linewidth=2.5, color=7, linestyle=:dashdot,
		label="vdW isotherm at $T1 K"
	)
	
	plot!([Vlsat1, Vvsat1], [psat1, psat1]/1e6,
		 linewidth=2, color=:black, linestyle=:dash,
		label="pressure construction"
	)
	
	Vroots = real.(cubic_volume(model, psat1, T1))
	scatter!(Vroots, repeat([psat1], 3)/1e6, label="volume roots", markersize=5.5, markershape=:diamond, color=4)
end

# ╔═╡ 8ef607ea-4c80-4da6-a209-ce34cf6fb55f
md"""
Under saturated conditions, the middle root never has physical meaning. Note also the line marked **pressure construction**. For a pure saturated fluid, an isotherm has **constant temperature**. This is not typically captured by an equation of state, so when designing software, care should be taken that calculations of pressure are physical and correct.

Let's now solve the van der Waals equation for the volume roots and see if our answers make sense.
"""

# ╔═╡ 49abc7bd-89b9-4926-9f6e-c2d1dd09a3e0
"""
	cubic_volume(model::ABCubicModel, p, T)

Solves a cubic equation of state for all volume roots, real or complex. Returns a vector of all three roots.
"""
function cubic_volume(model::ABCubicModel, p, T)
	Tc, pc, _ = crit_pure(model)

	a = 27/64 * (R*Tc)^2/pc # a parameter
	b = 1/8 * (R*Tc)/pc # b parameter

	# Rearranged cubic parameters
	A = a*p/(R*T)^2
	B = b*p/(R*T)

	poly = [-A*B, A, -(1+B), 1.0] # Polynomial coefficients
	Zvec = roots(poly) # Solve polynomial for Z
	Vvec = Zvec.*(R*T/p) # Transform to volume
	return Vvec
end

# ╔═╡ 94bbc71e-c79b-416c-a059-3e804d4fc107
begin
	cubic_model = vdW(["carbon dioxide"])
	p = 50e5
	T = 273.15
	Vvec = cubic_volume(cubic_model, p, T)
end

# ╔═╡ 78ac6273-5029-4aa0-9e44-caacab4e40ef
md"""
Now we have that working, we can see we have 3 real roots. How do we know which one to choose? We know that the smallest root is _liquidlike_, the largest root is _vapourlike_, and that the middle root has no physical meaning. From our physical knowledge of hydrogen sulfide, we can tell that it should be a gas and so we should take the vapourlike root, but this isn't something we can apply rigorously or put into our code.

In many situations the cubic equation will have imaginary roots, which are always unphysical and should be discarded. However, most of the time a decision between real roots has to be made. To do this, we will introduce the **gibbs free energy**, as we know that the phase with the lowest gibbs free energy (or chemical potential) will by the stable phase at the given conditions.

To evaluate the chemical potential, we can use

```
VT_chemical_potential(model, V, T)
```
"""

# ╔═╡ b4a5d7a8-c8b8-4e08-85d6-4eff52eccfe0
begin
	μ(V) = VT_chemical_potential(cubic_model, V, T)

	Vvec_real = real.(Vvec[abs.(imag.(Vvec)) .< eps()]) # Filter out volume roots with imaginary components below machine precision
	μvec = μ.(Vvec_real)
	μ_show = [round(i[1]; sigdigits=5) for i in μvec]
end

# ╔═╡ 6a497f80-2d63-497b-a273-82df3ca84c47
md"""
We can see that our lowest chemical potential is given by our third root, showing this is the most stable root and that it is the correct volume for the system.

We can compare our answer for $V$ to the result calculated by Clapeyron, using 

```
volume(model, p, T)
```
"""

# ╔═╡ 48deaba2-758b-4658-8c7b-2e07add0d2d6
V_cubic = Vvec_real[findmin(μvec)[2]]

# ╔═╡ 7ad14533-d7e6-4b44-9764-30dbdbac19f9
V_cubic ≈ volume(cubic_model, p, T)

# ╔═╡ f4ab1c61-1f6d-4a4c-8b0b-cf7c14f3d096
md"""
and see that we have chosen the correct root!
"""

# ╔═╡ 11bd73c1-c745-4d30-adc0-19209e0c0c82
md"""
## Statistical Associating Fluid Theory (SAFT)

With equations of state based off of SAFT (e.g. ogSAFT, SAFTVRMie), we have an expression for the residual helmholtz free energy. This is expressed as a sum of different contributions,

$$a^\mathrm{res} = a^\mathrm{seg} + a^\mathrm{chain} + a^\mathrm{assoc}~.$$

As this is already implemented in Clapeyron, all we need to know to use these equations of state is that we have an expression

$$a^\mathrm{res} = f(V,T)~.$$

One way to obtain an expression we could solve for the volume at a specified pressure and temperature would be to take the partial derivative of $a$ to express this as a nonlinear equation

$$\left(\frac{\partial a^\mathrm{res}}{\partial V}\right)_T = -p~.$$

This can then be rearranged to 

$$f(V,T,p) = \left(\frac{\partial a^\mathrm{res}(V,T)}{\partial V}\right)_T + p = 0\tag{1}$$

giving us a non-linear root-finding problem.
"""

# ╔═╡ d0cfc031-3153-4ac5-9b50-1fba2729e9f4
md"""
An alternative approach is to begin with the definition of isothermal compressibility

$$\beta = -\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_T$$

which can be integrated to

$$\beta\cdot(p_2 - p_1) = \ln(V_1)-\ln(V_2)$$
$$\exp(\beta\cdot(p_2 - p_1)) = \frac{V_1}{V_2}$$
$$V_1 = V_2\exp(\beta\cdot(p_2 - p_1))~.$$

If we take $p_2$ as the specification pressure, and $p_1$ a function of $V_2$, then we have obtained a formula we can use to iterate to convergence.

$$V_{i+1} = V_i \exp(\beta(V_i) \cdot (p^\mathrm{spec} - p(V_i)))$$

Where every variable is evaluated at the specification temperature, $T^\mathrm{spec}$.

To increase numerical stability this is then moved to log-space

$$\ln V_{i+1} = \ln V_i \cdot β(V_i)\cdot(p^\mathrm{spec} - p(V_i))~.\tag{2}$$

In practice, equation (2) converges faster than directly solving equation (1). [^1]
"""

# ╔═╡ 8ecab7d3-be38-4733-b02f-9b00d5e75bd1
md"""
We now have a relation that will converge to a volume root of our equation via **successive substitution**, but how do we go about generating initial guesses?

### Initial  guesses

To generate liquid-like initial guesses for SAFT equations, we're going to use a method based off of the packing fraction. This is defined as

$$\eta = \frac{\frac{3\pi}{3}\cdot \left(\frac{\sigma}{2}\right)^3\cdot N_A}{V}$$

where $\sigma$ is the segment size and $N_A$ is Avogadro's number.
"""

# ╔═╡ 2c4e32b3-0eee-427b-bace-e723952d2e5b
md"""
The packing fraction can visually be seen as the point at which all space in a given volume is taken up by fluid molecules:
"""

# ╔═╡ 5b8815a8-a079-4b8f-a618-2d48269812b8
@htl("""<center><img src="https://raw.githubusercontent.com/lucpaoli/introduction-to-computational-thermodynamics/main/Part%203%20-%20Solving%20the%20problems/assets/packing_fraction.png" height="190"></center>""")

# ╔═╡ aa58fba4-9450-4aa2-8679-5dc953ff730e
md"""
As we take the limit of the packing fraction to one, 

$$1 = \frac{\frac{\pi}{6}\cdot N_A\cdot \sigma^3}{V}$$

Giving us the expression for our volume guess as

$$V_0^\mathrm{liq} = \frac{\pi}{6}\cdot N_A \cdot \sigma^3~.$$
"""

# ╔═╡ 7cfbb284-72ae-4d7f-ad5a-c98dd4ad2f97
md"""
For the vapour-like initial guesses we can use the ideal gas equation

$$V^\mathrm{vap} = \frac{nRT}{p}~.$$

It would be possible to improve this by including further terms of the Virial equation, for example

$$Z^\mathrm{vap} = 1 + \frac{B(T)}{V}$$

but this isn't necessary for now, as the sequence defined by equation (2) generally converges very quickly.
"""

# ╔═╡ 852f0913-187a-4a88-b62e-f7597c6663dc
md"""
It is typical for PCSAFT to have three roots, though it can have up to five! Five roots can usually only occur in unphysical scenarios, such as the example in [^2] of a decane isotherm at 135 K. Generally, this does not prove too much of an issue, though it emphasises the need for reasonable initial guesses, as convergence to unexpected and unphysical volume roots or critical points does become a possibility.
"""

# ╔═╡ 07c9b193-3ca8-47e7-91ad-0492e2dddf2c
md"""
T =
"""

# ╔═╡ d7a60eae-0393-4b11-b1e1-faec368d324b
@bind T_plt2 PlutoUI.Slider(250u"K":1u"K":325u"K", show_value=true, default=295u"K")

# ╔═╡ 68cad9ec-8f72-41f1-8665-3b0fb87147de
let
	model = PCSAFT(["carbon dioxide"])
	
	Tcrit, pcrit, vcrit = crit_pure(model)
	Tsat = LinRange(200.0, 0.9999Tcrit, 500)

	psat = zeros(length(Tsat))
	Vlsat = zeros(length(Tsat))
	Vvsat = zeros(length(Tsat))
	for (i, T) in enumerate(Tsat)
		(psat[i], Vlsat[i], Vvsat[i]) = saturation_pressure(model, T)
	end

	psat = psat./1e6
	pcrit = pcrit/1e6
	gr()
	plt = plot(
		title="pv diagram for CO₂ using the PCSAFT EoS",
		ylabel="p (MPa)", xlabel="v (m³/mol)",
		framestyle=:box, tick_direction=:out, grid=:off,
		xscale=:log,
		ylim = (minimum(psat), 1.1pcrit)
	)
	
	plot!(
		Vlsat, psat,
		linewidth=2, color=1, linestyle=:solid,
		label="Saturation curve"
	)
	plot!(
		Vvsat, psat,
		linewidth=2, color=1, linestyle=:solid,
		label=""
	)
	scatter!([vcrit], [pcrit], label="critical point", markersize=5.5, color=3)

	T1 = ustrip(T_plt2)
	psat1, Vlsat1, Vvsat1 = saturation_pressure(model, T1)
	
	plot!(
		v -> pressure(model, v, T1)/1e6,
		10 .^ range(log10(minimum(Vlsat)), log10(maximum(Vvsat)), 1000),
		linewidth=2.5, color=7, linestyle=:dashdot,
		label="PCSAFT isotherm at $T1 K"
	)
	
	plot!([Vlsat1, Vvsat1], [psat1, psat1]/1e6,
		 linewidth=2, color=:black, linestyle=:dash,
		label="pressure construction"
	)
	
	# scatter!(Vroots, repeat([psat1], 3)/1e6, label="volume roots", markersize=5.5, markershape=:diamond, color=4)
end

# ╔═╡ 87867c7c-101c-4534-b6e7-d3ebc8b81a47
md"""
### Implementation

To implement this, the functions we'll need are

```julia
β = VT_isothermal_compressibility(model, V, T)
p = pressure(model, V, T)
 
```

The ```volume_guess``` function for the liquid phase directly indexes the model object to obtain the sigma and segment values. Because only SAFT type models have these parameters, it is important we restrict our function to only accept the correct types. We do this with the ```::SAFTModel``` syntax in our function definition.
"""

# ╔═╡ dbb341dd-e535-470f-b43a-b28ed13a7af9
"""
	volume_guess(model, p, T, phase)

Generates initial guessses for the a volume solver. The liquid phase initial guess is based off of the packing fraction limit, and the vapour phase guess is based off of the ideal gas equation.
"""
function volume_guess(model::SAFTModel, p, T, phase)
	if phase == :liquid
		# Extract parameters 
		σ = model.params.sigma.diagvalues[1]
		seg = model.params.segment.values[1]
		V0 = 1.25 * π/6 * N_A * σ^3 * seg
	elseif phase == :vapour
		V0 = R*T/p
	else
		@error "Invalid phase specification, $phase. Specify phase as either ```:liquid``` or ```:vapour```"
	end
	return V0
end

# ╔═╡ ad443d16-38d5-4a39-b9a8-28576e40d52e
md"""
In this implementation of ```SAFT_volume```, ```abstol``` and ```maxiters``` are **keyword arguments**. These are optional when calling the function, and have given default values. For example, you could either call

```
SAFT_volume(model, p, T, phase)
```
or for higher precision
```
SAFT_volume(model, p, T, phase; abstol=1e-10)
```

you can read more about keyword arguments [here](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments).
"""

# ╔═╡ 27ac6272-6d1e-4c8d-8660-5d7c049d0285
"""
	SAFT_volume(model, p, T, phase; abstol=1e-9, maxiters=100)

Solves an equation of state for a volume root. The root converged to is chosen by the initial guess, which is specified by the phase argument. The phase can either be liquid, ```:liquid```, or vapour, ```:vapour```.
"""
function SAFT_volume(model, p, T, phase;
					abstol=1e-9,
					maxiters=100,
					V0=volume_guess(model, p, T, phase)
	)
	
	β(V) = VT_isothermal_compressibility(model, V, T)
	p₁(V) = pressure(model, V, T)

	V = V0
	lnV = log(V)
	test_norm = 1.0
	iters = 0
	while test_norm > abstol && iters < maxiters
		d = β(V) * (p₁(V) - p) # Calculate the iterative step
		lnV = lnV + d # Take the step

		# Copy previous iteration and value
		Vold = V
		V = exp(lnV)
		# Calculate convergence critera
		test_norm = abs(Vold - V)
	end
	
	if iters == maxiters
		@warn "Volume iteration failed to converge in $maxiters iterations"
	end
	return V
end

# ╔═╡ 04299697-4527-4fc7-a8bc-0d1c1a10bea5
md"""
Now our functions have been defined we can define our model, then call the function defined above to calculate the volume for a given phase.
"""

# ╔═╡ 8da908bb-13b6-48e0-9023-506c9bf2a1ee
begin
	SAFT_model = PCSAFT(["carbon dioxide"])
	V_SAFT = SAFT_volume(SAFT_model, p, T, :liquid)
end

# ╔═╡ fea36d6d-572d-4421-8616-ea59820c225e
md"""
We can again compare our answer to Clapeyron
"""

# ╔═╡ fc58d8f6-9647-4331-bebd-624ba7d75e7f
md"""
and we see that we've converged to the same answer!
"""

# ╔═╡ 123c84a4-f363-4569-ba96-408a376fd4cf
md"""
Note that if we don't know _a-priori_ which phase we should solve for, it's once again necessary to solve for both the liquid and vapour roots and evaluate the chemical potential to determine which is more stable.
"""

# ╔═╡ c9974fa6-6418-4c4e-bf4e-37bc97a03a40
md"""
### Convergence 

We can also look at the convergence of our volume solver method by plotting the performance of our successive substitution relation against the plot of the first objective function we derived.
"""

# ╔═╡ 5448c537-7ace-42d9-b2b6-addb1bfc8152
md"""
$$f(V,T,p) = \left(\frac{\partial a^\mathrm{res}(V,T)}{\partial V}\right)_T + p^\mathrm{spec}$$
"""

# ╔═╡ 83db06aa-ec89-40dc-b379-b63c8fce93f4
function SAFT_volume_trace(model, p, T, phase;
					abstol=1e-9,
					maxiters=100,
					V0=volume_guess(model, p, T, phase)
	)
	
	β(V) = VT_isothermal_compressibility(model, V, T)
	p₁(V) = pressure(model, V, T)

	V = V0
	lnV = log(V)
	Vtrack = [V]
	test_norm = 1.0
	iters = 0
	while test_norm > abstol && iters < maxiters
		d = β(V) * (p₁(V) - p) # Calculate the iterative step
		lnV = lnV + d # Take the step

		# Copy previous iteration and value
		Vold = V
		V = exp(lnV)
		push!(Vtrack, V)
		# Calculate convergence critera
		test_norm = abs(Vold - V)
	end
	
	if iters == maxiters
		@warn "Volume iteration failed to converge in $maxiters iterations"
	end
	return V, Vtrack
end;

# ╔═╡ 2f97def3-582d-4c83-a5b9-1c4a661a6142
let
	model = PCSAFT(["carbon dioxide"])
	β(V) = VT_isothermal_compressibility(model, V, T)
	p = 50e5
	T = 273.15
	# f_plt(V) = V*(1-exp(β(V)*(p - pressure(model, V, T))))
	f_plt(V) = (pressure(model, V/1e5, T) - p)/1e10

	V0 = volume_guess(model, p, T, :liquid)
	V, Vtrack = SAFT_volume_trace(model, p, T, :liquid)
	Vtrack = Vtrack[1:end-1]

	V0 = V0*1e5
	V = V*1e5
	Vtrack = Vtrack*1e5
	
	V_range = range(V0, 1.1V, 500)
	
	gr()
	plt = plot(
		title="",
		ylabel="f(V) / 10¹⁰", xlabel="v (m³/mol) / 10⁻⁵",
		framestyle=:box, tick_direction=:out, grid=:on,
		xlim = (0.95V_range[1], V_range[end]),
		dpi = 800
	)
	
	hline!([0.0], color=:black, label="")
	plot!(V_range, f_plt.(V_range),
		color = 1,
		linewidth=2,
		label="volume objective function"
	)
	scatter!(Vtrack, f_plt.(Vtrack),
		color = 2,
		markersize=4.5,
		label="iterations"
	)
	scatter!([V0], [f_plt(V0)],
		color = 3,
		markersize=4.5,
		label="V₀"
	)
	scatter!([V], [f_plt(V)],
		color = 4,
		markersize=4.5,
		label="V"
	)

	Vtrack = Vtrack[end-4:end]
	V_range = range(0.982Vtrack[1], 1.025Vtrack[end], 500)

	x1 = V_range[1]
	x2 = V_range[end]
	x3 = 3.3455
	x4 = 5.12
	y1 = -5e8/1e10
	y2 = 5e8/1e10
	y3 = 0.895*5e9/1e10

	plot!([x1, x2], [y1, y1], color=:black, label="")
	plot!([x1, x2], [y2, y2], color=:black, label="")
	plot!([x1, x1], [y1, y2], color=:black, label="")
	plot!([x2, x2], [y1, y2], color=:black, label="")
	plot!([x1, x3], [y2, y3], color=:black, label="")
	plot!([x2, x4], [y2, y3], color=:black, label="")

	plot!(subplot=2,
		title="",
		framestyle=:box, tick_direction=:out, grid=:on,
		xticks=(4.4:0.1:4.9),
		# xlim = (0.95V_range[1], V_range[end]),
		# yticks=false, xticks=false,
		xlim = (V_range[1], V_range[end]),
		inset = (1, bbox(0.03, 0.03, 0.5, 0.5, :center, :right)),
	)
	
	hline!(subplot=2, [0.0], color=:black, label="")
	plot!(subplot=2, V_range, f_plt.(V_range),
		color = 1,
		linewidth=2,
		label=""
	)
	scatter!(subplot=2, Vtrack, f_plt.(Vtrack),
		color = 2,
		markersize=4.5,
		label=""
	)
	scatter!(subplot=2, [V], [f_plt(V)],
		color = 4,
		markersize=4.5,
		label=""
	)
	

	plt
end

# ╔═╡ aa6a0a8c-20a2-43bd-8dc7-682ce36b1ab8
md"""
We can see fairly rapid convergence, though it does become clear that our initial guess is far from perfect!
"""

# ╔═╡ 73acffd2-3f5e-4a6b-b135-1e128eac0577
md"""
## Helmholtz-Explicit (e.g. GERG, IAPWS-95)
Because the volume solver we developed for SAFT is a non-linear solver making no assumptions about the form of the equation of state other than that it can be expressed explicitly in terms of the Helmholtz free energy, we can actually use that solver directly with any other equation of state!
The only issue encountered with this is we need to re-think our initial guesses. This is generally dependent on the model used in question.
"""

# ╔═╡ b4c2baf6-5433-45e0-b191-b43c38f346ef
md"""
Helmholtz-Explicit equations of state now have more than 3 volume roots within the saturated region. This introduces some additional risk when solving for the volume for converging to the incorrect root, but for these equations of state there are usually well-defined methods for generating initial guesses, such as a known solution to the IAPWS-95 saturation curve, or a select reference fluid for the GERG-2004 model.
"""

# ╔═╡ 8af302d2-371e-457d-a0de-06819d01df92
@htl("""
T<sub>1</sub> =
""")

# ╔═╡ b7369d25-7286-4245-9980-d0d4ce831947
@bind T_plt3 PlutoUI.Slider(500u"K":5u"K":700u"K", show_value=true, default=630u"K")

# ╔═╡ 7512fa2f-b519-4f7a-8c7e-ca56810db2be
@htl("""
T<sub>2</sub> =
""")

# ╔═╡ 4a5e5154-adb7-4a12-8bdd-39d6ef55b6c1
@bind T_plt4 PlutoUI.Slider(100u"K":5u"K":200u"K", show_value=true, default=175u"K")

# ╔═╡ b156c581-7828-45e9-a683-4d925216aed1
let
	p1 = let
		model = IAPWS95()
		
		Tcrit, pcrit, vcrit = crit_pure(model)
		Tsat = LinRange(550.0, 0.99999Tcrit, 500)
	
		psat = zeros(length(Tsat))
		Vlsat = zeros(length(Tsat))
		Vvsat = zeros(length(Tsat))
		for (i, T) in enumerate(Tsat)
			(psat[i], Vlsat[i], Vvsat[i]) = saturation_pressure(model, T)
		end
	
		psat = psat./1e6
		pcrit = pcrit/1e6
		gr()
		plt = plot(
			title="H₂O using IAPWS-95 EoS",
			ylabel="p (MPa)", xlabel="v (m³/mol)",
			framestyle=:box, tick_direction=:out, grid=:off,
			xscale=:log, legend=:bottomleft,
			ylim = (minimum(psat), 1.1pcrit)
		)
		
		plot!(
			Vlsat, psat,
			linewidth=2, color=1, linestyle=:solid,
			label="saturation curve"
		)
		plot!(
			Vvsat, psat,
			linewidth=2, color=1, linestyle=:solid,
			label=""
		)
		scatter!([vcrit], [pcrit], label="critical point", markersize=5.5, color=3)
	
		T1 = ustrip(T_plt3)
		psat1, Vlsat1, Vvsat1 = saturation_pressure(model, T1)
		
		plot!(
			v -> pressure(model, v, T1)/1e6,
			10 .^ range(log10(minimum(Vlsat)), log10(maximum(Vvsat)), 1000),
			linewidth=2.5, color=7, linestyle=:dashdot,
			label="isotherm at $T1 K"
		)
		
		plot!([Vlsat1, Vvsat1], [psat1, psat1]/1e6,
			 linewidth=2, color=:black, linestyle=:dash,
			label="pressure construction"
		)
	end
	p2 = let
		model = GERG2008(["methane"])
		
		Tcrit, pcrit, vcrit = crit_pure(model)
		Tsat = LinRange(140.0, 0.99999Tcrit, 500)
	
		psat = zeros(length(Tsat))
		Vlsat = zeros(length(Tsat))
		Vvsat = zeros(length(Tsat))
		for (i, T) in enumerate(Tsat)
			(psat[i], Vlsat[i], Vvsat[i]) = saturation_pressure(model, T)
		end
	
		psat = psat./1e6
		pcrit = pcrit/1e6
		gr()
		plt = plot(
			title="CH₄ using GERG-2004",
			ylabel="p (MPa)", xlabel="v (m³/mol)",
			framestyle=:box, tick_direction=:out, grid=:off,
			xscale=:log, legend=:bottomleft,
			ylim = (minimum(psat), 1.1pcrit)
		)
		
		plot!(
			Vlsat, psat,
			linewidth=2, color=1, linestyle=:solid,
			label="saturation curve"
		)
		plot!(
			Vvsat, psat,
			linewidth=2, color=1, linestyle=:solid,
			label=""
		)
		scatter!([vcrit], [pcrit], label="critical point", markersize=5.5, color=3)
	
		T1 = ustrip(T_plt4)
		psat1, Vlsat1, Vvsat1 = saturation_pressure(model, T1)
		
		plot!(
			v -> pressure(model, v, T1)/1e6,
			10 .^ range(log10(minimum(Vlsat)), log10(maximum(Vvsat)), 1000),
			linewidth=2.5, color=7, linestyle=:dashdot,
			label="isotherm at $T1 K"
		)
		
		plot!([Vlsat1, Vvsat1], [psat1, psat1]/1e6,
			 linewidth=2, color=:black, linestyle=:dash,
			label="pressure construction"
		)
	end

	plot(p1, p2, layout = @layout [a b])
end

# ╔═╡ 4acb1393-030f-4cab-a765-f8de5a75893b
html"<br><br><br><br><br><br><br><br><br><br><br><br>"

# ╔═╡ d9835e4a-e64e-4b3a-8c3c-f9d3766b23b9
md"""
# Footnotes
[^1]: The recurrence relation formed by equation (2) turns out to be a Newton step towards the solution of equation (1).
[^2]: This is shown in Fig. 7 of:
"""

# ╔═╡ 2d294f2c-2b68-4dbb-8f28-1d57e113f674
DOI("10.1016/j.fluid.2010.03.041")

# ╔═╡ d0b2f6bb-7539-4dda-adc9-acc2ce9cca4a
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]));

# ╔═╡ 8fe83aab-d193-4a28-a763-6420abcbb176
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]));

# ╔═╡ 94caf041-6363-4b38-b2c2-daaf5a6aecf1
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]));

# ╔═╡ 217956f7-f5f5-4345-8642-7736dc4321d7
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));

# ╔═╡ dbe0cb67-b166-40b6-aeaf-a2e2d6ca4c87
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."];

# ╔═╡ f67c10e6-8aa1-4eed-9561-b629fa8ac91b
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]));

# ╔═╡ 54de307b-64c3-4bed-9c91-d8ab7d3ad815
if V_cubic ≈ volume(cubic_model, p, T)
	correct(md"You converged within machine precision!")
else
	almost(md"Not quite there, you're off by δ = $(V_SAFT - volume(SAFT_model, p, T))")
end

# ╔═╡ 2d740e80-c450-46c2-9086-75f4ce23ce26
if V_SAFT ≈ volume(SAFT_model, p, T)
	correct()
else
	almost(md"Not quite there, you're off by δ = $(V_SAFT - volume(SAFT_model, p, T))")
end

# ╔═╡ 970bb661-c959-4f0c-a1d6-50f655b80ef8
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]));

# ╔═╡ 156674ce-b49e-44b6-8182-7f8da0a394af
if !@isdefined(Vvec)
	not_defined(:Vvec)
else
	if Vvec[findmin(μ_show)[2]] ≈ volume(cubic_model, p, T)
		correct()
	end
end

# ╔═╡ 808c11cb-f930-4fd6-b827-320e845a47a7
function reduce_complex(x)
	x2 = Vector{Union{Float64, ComplexF64}}(x)
	x2[abs.(imag.(x)) .< eps()] .= map(x -> real(x), x[abs.(imag.(x)) .< eps()])
	map!(x -> round(x; sigdigits=3), x2, x2)
	# x[abs.(imag.(x)) .< eps()] .= real(x[abs.(imag.(x)) .< eps()])
	return x2
end;

# ╔═╡ 90baae39-478b-493c-ac23-4fadca9c3698
begin
	try
		Vvec_show = reduce_complex(Vvec)
		if length(Vvec_show) == 3
			@htl("""
			<table>
			  <tr>
			    <th>Root</th>
				<th>V (m³)</th>
			  </tr>
			  <tr>
			    <td>1</td>
			    <td>$(Vvec_show[1])</td>
			  </tr>
			  <tr>
			    <td>2</td>
			    <td>$(Vvec_show[2])</td>
			  </tr>
			  <tr>
			    <td>3</td>
			    <td>$(Vvec_show[3])</td>
			  </tr>
			</table>
			""")
		elseif length(Vvec_show) == 1
			@htl("""
			<table>
			  <tr>
			    <th>Root</th>
				<th>V (m³)</th>
			  </tr>
			  <tr>
			    <td>1</td>
			    <td>$(Vvec_show[1])</td>
			  </tr>
			</table>
			""")
		else
			@error ""
		end
	catch
		@htl("""
		<table>
		  <tr>
		    <th>Root</th>
			<th>V (m³)</th>
		  </tr>
		  <tr>
		    <td>1</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>2</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>3</td>
		    <td>?</td>
		  </tr>
		</table>
		""")
	end
end

# ╔═╡ 0c03142f-dd0f-4769-9757-a03b72049bf3
begin
	try
		# μ_show = map(x -> round(x[1]; sigdigits=5), μvec)
		Vvec_show = reduce_complex(Vvec)
		if length(μ_show) == 3
			@htl("""
			<table>
			  <tr>
			    <th>Root</th>
				<th>μ</th>
				<th>V</th>
			  </tr>
			  <tr>
			    <td>1</td>
			    <td>$(μ_show[1])</td>
			    <td>$(Vvec_show[1])</td>
			  </tr>
			  <tr>
			    <td>2</td>
			    <td>$(μ_show[2])</td>
			    <td>$(Vvec_show[2])</td>
			  </tr>
			  <tr>
			    <td>3</td>
			    <td>$(μ_show[3])</td>
			    <td>$(Vvec_show[3])</td>
			  </tr>
			</table>
			""")
		elseif length(μ_show) == 1
			@htl("""
			<table>
			  <tr>
			    <th>Root</th>
				<th>μ</th>
			  </tr>
			  <tr>
			    <td>1</td>
			    <td>$(μ_show[1])</td>
			  </tr>
			</table>
			""")
		else
			@error ""
		end
	catch e
		@htl("""
		<table>
		  <tr>
		    <th>Root</th>
			<th>V (m³)</th>
		  </tr>
		  <tr>
		    <td>1</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>2</td>
		    <td>?</td>
		  </tr>
		  <tr>
		    <td>3</td>
		    <td>?</td>
		  </tr>
		</table>
		""")
		e
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clapeyron = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PolynomialRoots = "3a141323-8675-5d76-9d11-e1df1406c778"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
Clapeyron = "~0.3.6"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Optim = "~1.7.0"
Plots = "~1.30.1"
PlutoUI = "~0.7.39"
PolynomialRoots = "~1.0.0"
Roots = "~2.0.1"
ShortCodes = "~0.3.3"
Unitful = "~1.11.0"
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

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d618d3cf75e8ed5064670e939289698ecf426c7f"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.12"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clapeyron]]
deps = ["BlackBoxOptim", "CSV", "DiffResults", "FillArrays", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "NLSolvers", "PackedVectorsOfVectors", "PositiveFactorizations", "Roots", "Scratch", "SparseArrays", "StaticArrays", "Tables", "ThermoState", "UUIDs", "Unitful"]
git-tree-sha1 = "842e3003389750193c2fbc7bdd62c5ba9f2f3cc5"
uuid = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
version = "0.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

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

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

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

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

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
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ee13c773ce60d9e95a6c6ea134f25605dce2eda3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.13.0"

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

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

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
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

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
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

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
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

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
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

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
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

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
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

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
git-tree-sha1 = "7f4869861f8dac4990d6808b66b57e5a425cfd99"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.13"

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
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "2402dffcbc5bb1631fb4f10cb5c3698acdce29ea"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.30.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

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
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

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
git-tree-sha1 = "30e3981751855e2340e9b524ab58c1ec85c36f33"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

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
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2bbd9f2e40afd197a1379aef05e0d85dba649951"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.7"

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
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

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
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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
# ╟─2f3a7b9d-3dac-4e12-abb1-ce09cef36e93
# ╟─663f44e1-6f7c-490c-9f7a-446f94a63c2a
# ╟─2807debe-5c79-40a7-acf3-947b288449d7
# ╟─591b6a0b-548b-44f4-8415-16b8aa219c0b
# ╟─fee14db9-d42d-4a6b-8ef3-a539421be776
# ╟─8ef607ea-4c80-4da6-a209-ce34cf6fb55f
# ╠═49abc7bd-89b9-4926-9f6e-c2d1dd09a3e0
# ╠═94bbc71e-c79b-416c-a059-3e804d4fc107
# ╟─90baae39-478b-493c-ac23-4fadca9c3698
# ╟─156674ce-b49e-44b6-8182-7f8da0a394af
# ╟─78ac6273-5029-4aa0-9e44-caacab4e40ef
# ╠═b4a5d7a8-c8b8-4e08-85d6-4eff52eccfe0
# ╟─0c03142f-dd0f-4769-9757-a03b72049bf3
# ╟─6a497f80-2d63-497b-a273-82df3ca84c47
# ╠═48deaba2-758b-4658-8c7b-2e07add0d2d6
# ╠═7ad14533-d7e6-4b44-9764-30dbdbac19f9
# ╟─54de307b-64c3-4bed-9c91-d8ab7d3ad815
# ╟─f4ab1c61-1f6d-4a4c-8b0b-cf7c14f3d096
# ╟─11bd73c1-c745-4d30-adc0-19209e0c0c82
# ╟─d0cfc031-3153-4ac5-9b50-1fba2729e9f4
# ╟─8ecab7d3-be38-4733-b02f-9b00d5e75bd1
# ╟─2c4e32b3-0eee-427b-bace-e723952d2e5b
# ╟─5b8815a8-a079-4b8f-a618-2d48269812b8
# ╟─aa58fba4-9450-4aa2-8679-5dc953ff730e
# ╟─7cfbb284-72ae-4d7f-ad5a-c98dd4ad2f97
# ╟─852f0913-187a-4a88-b62e-f7597c6663dc
# ╟─07c9b193-3ca8-47e7-91ad-0492e2dddf2c
# ╟─d7a60eae-0393-4b11-b1e1-faec368d324b
# ╟─68cad9ec-8f72-41f1-8665-3b0fb87147de
# ╟─87867c7c-101c-4534-b6e7-d3ebc8b81a47
# ╠═dbb341dd-e535-470f-b43a-b28ed13a7af9
# ╟─ad443d16-38d5-4a39-b9a8-28576e40d52e
# ╠═27ac6272-6d1e-4c8d-8660-5d7c049d0285
# ╟─04299697-4527-4fc7-a8bc-0d1c1a10bea5
# ╠═8da908bb-13b6-48e0-9023-506c9bf2a1ee
# ╟─fea36d6d-572d-4421-8616-ea59820c225e
# ╟─2d740e80-c450-46c2-9086-75f4ce23ce26
# ╟─fc58d8f6-9647-4331-bebd-624ba7d75e7f
# ╟─123c84a4-f363-4569-ba96-408a376fd4cf
# ╟─c9974fa6-6418-4c4e-bf4e-37bc97a03a40
# ╟─5448c537-7ace-42d9-b2b6-addb1bfc8152
# ╟─2f97def3-582d-4c83-a5b9-1c4a661a6142
# ╟─83db06aa-ec89-40dc-b379-b63c8fce93f4
# ╟─aa6a0a8c-20a2-43bd-8dc7-682ce36b1ab8
# ╟─73acffd2-3f5e-4a6b-b135-1e128eac0577
# ╟─b4c2baf6-5433-45e0-b191-b43c38f346ef
# ╟─8af302d2-371e-457d-a0de-06819d01df92
# ╟─b7369d25-7286-4245-9980-d0d4ce831947
# ╟─7512fa2f-b519-4f7a-8c7e-ca56810db2be
# ╟─4a5e5154-adb7-4a12-8bdd-39d6ef55b6c1
# ╟─b156c581-7828-45e9-a683-4d925216aed1
# ╟─4acb1393-030f-4cab-a765-f8de5a75893b
# ╟─d9835e4a-e64e-4b3a-8c3c-f9d3766b23b9
# ╟─2d294f2c-2b68-4dbb-8f28-1d57e113f674
# ╟─d0b2f6bb-7539-4dda-adc9-acc2ce9cca4a
# ╟─8fe83aab-d193-4a28-a763-6420abcbb176
# ╟─94caf041-6363-4b38-b2c2-daaf5a6aecf1
# ╟─217956f7-f5f5-4345-8642-7736dc4321d7
# ╟─dbe0cb67-b166-40b6-aeaf-a2e2d6ca4c87
# ╟─f67c10e6-8aa1-4eed-9561-b629fa8ac91b
# ╟─970bb661-c959-4f0c-a1d6-50f655b80ef8
# ╟─808c11cb-f930-4fd6-b827-320e845a47a7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
