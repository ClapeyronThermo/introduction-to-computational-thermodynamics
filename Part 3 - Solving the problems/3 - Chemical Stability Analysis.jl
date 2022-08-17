### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a598b2d2-f15c-11ec-10ed-a15ca6a73a60
begin
	# Specific functions from Clapeyron
    using Clapeyron: wilson_k_values, R̄
	const R = R̄
	using Clapeyron, ForwardDiff, LinearAlgebra, NLsolve, Optimization, OptimizationOptimJL
	using Statistics
	using HypertextLiteral
    using LaTeXStrings, Plots, PlutoUI
	using ShortCodes
	# For 3D plots
	import PlotlyJS
	# For ternary contour plots
	# import GMT
end

# ╔═╡ 04089557-9381-4bbb-9882-45b5a685694b
md"""
# Section 3.3 - Stability Analysis

Stability analysis deals with the question "**does a phase split occur?**". This can be very important as it allows us to determine whether further calculation is necessary, or if we can simply evaluate the phase with the lowest Gibbs free energy ($G$). It is a key part of multiphase flash algorithms, which you will see in section 3.5, as well as providing good initial guesses to our two-phase flash, which we will develop in section 3.4.

## Gibbs tangent-plane condition

The fundamental description of stability is the Gibbs tangent-plane. We know a system always exists at the minima on the Gibbs free energy surface, so when a tangent plane with a lower $G$ can be constructed, we know that a phase split will occur, as this multiple phase solution will be more stable than the single phase.

For a binary mixture this can be quite easy to visualise. In the example of methane and hydrogen sulfide below you can see the mixture will split into two phases, shown by the intersection of the "equilibrium tangent plane" with the gibbs free energy of mixing surface.
"""

# ╔═╡ cc88e502-8b06-44fc-b888-3bed9cf1b683
# fig3

# ╔═╡ 87bb11a1-d8f7-4b51-b67a-2cfeef4ad039
md"""
As we move to mixtures with more components - such as the octane, ethane, and propane ternary mixture represented below - it can become a lot harder to tell. For mixtures of many components, this can become impossible to fully visualise.
"""

# ╔═╡ f1c2c57d-0d05-48d4-ae6d-a015001d1888
md"""
There is therefore a need for a mathematical description of phase stability that we can apply to any number of components and phases.

This description can be derived by considering the fact the Gibbs free energy being at a **global minimum**. This requires that the formation of any new phase must increase $G$.

The change in Gibbs free energy upon the formation of $\delta e$ of a new phase with composition $\mathbf x$ from a prexisting mixture with composition $\mathbf z$ can be written as

$$\delta G = \delta e \sum x_i(\mu_i(\mathbf x) - \mu_i(\mathbf z))~.$$

For a mixture to be stable, we must therefore have that 

$$\delta G \geq 0 \implies \sum x_i (\mu_i(\mathbf x) - \mu_i(\mathbf z)) \geq 0$$

Physically, this represents the distance from the gibbs free energy surface at overall composition $\mathbf z$ to a tangent plane constructed at composition $\mathbf x$. Because of this we call this function the **tangent plane distance** function, or

$$TPD(\mathbf x) = \sum x_i (\mu_i(\mathbf x) - \mu_i(\mathbf z))$$

and we can see that if $TPD \geq 0$ for all possible compositions $\mathbf x$, then our mixture must be stable. Another way of looking at this is is that if we evaluate every minima of $TPD$, and they are all positive, then the mixture is stable.

The $TPD$ function can be seen visualised below:
"""

# ╔═╡ 81a77642-52ef-4836-9055-5059332f72d8
md"""
## Unconstrained formulation

We now have a function describing whether a mixture is stable or not, but how do we go about implementing this?

If we were to attempt to naively program this in, we would end up with a poorly scaled constrained minimisation problem, requiring all our mole fractions sum to one. While this is possible to solve, it is more complicated than necessary.

> $$\min TPD(p^\mathrm{spec}, T^\mathrm{spec}, \mathbf z^\mathrm{spec}, \mathbf x)$$
> subject to
>
> $$\begin{gather}
> 0 \leq w_i \leq 1\quad\forall i \in [1, C]\\
> \sum w_i = 1
> \end{gather}$$

The first change we make is expressing this in terms of the fugacity coefficients, as typically these are easier to work with. While this isn't technically true when using Clapeyron, it is still a convention that we will follow. To do this, we rewrite our chemical potential in terms of our fugacity coefficients

$$\mu_i = \mu_i^* + RT\ln\frac{f_i}{P^\mathrm{ref}} = \mu_i^* + RT(\ln x_i + \ln \varphi_i)$$

noting that $P^\mathrm{ref}$ is taken as one. Our function is now 

$$TPD(\mathbf x) = \sum x_i (\ln x_i + \ln \varphi_i(\mathbf x) - \ln x_i - \ln \varphi_i(\mathbf z))$$

and we then define a "helper variable", $d_i$, to simplify our expression

$$d_i = \ln z_i - \ln\varphi_i(\mathbf z)$$

To better scale our problem we use the same $RT$ scaling factor as before

$$tpd = \frac{TPD}{RT} = \sum x_i (\ln x_i + \ln φ_i(\mathbf x) - d_i)$$
where

$$d_i = \ln z_i + \ln \varphi_i(\mathbf z)$$

Finally, we remove the mass balance constraints by reformulating changing it to be a function of mole numbers, $\mathbf X$,

$$tm(\mathbf X) = 1 + \sum_i X_i (\ln X_i + \ln \varphi_i(\mathbf X) - d_i - 1)~.$$

Because of this change $tm$ no longer describes the tangent plane distance, however it can be shown that [1]

$$\begin{gather}
tm < 0 \iff TPD < 0\\
\min tm \iff \min TPD
\end{gather}$$

meaning $tm$ can be used identically to $TPD$ in the context of stability analysis.

> $$\min tm(p^\mathrm{spec}, T^\mathrm{spec}, \mathbf z^\mathrm{spec}, \mathbf X)$$
> subject to
>
> $$0 \leq X_i \quad\forall i \in [1, C]$$

To satisfy the the final constraint, we use logs. To do this we change our iteration variable from $\mathbf X$ to $\log \mathbf X$

> $$\min tm(p^\mathrm{spec}, T^\mathrm{spec}, \mathbf z^\mathrm{spec}, 10^{\log\mathbf X})$$

This method is applicable to any number of components, though is easiest to visualise for a binary mixture.
"""

# ╔═╡ 1db67ed3-53cf-42a9-a8a0-be3e2a2dc537
md"""
## Implementation

Now we're familiar with our problem, and we have an objective function let's implement it.

This will be split into three stages:

###### 1. State and model specification:
* Specify $p, T, \mathbf{z}$
###### 2. Generate initial guesses
* Use a correlation to obtain $\mathbf{x}_0$
###### 3. Minimise our objective function $tm$
* Use an optimisation algorithm to minimise the tangent plane distance

### 1. State and model specification

We're going to use a predictive cubic, EPPR78, to capture nonidealities with the binary interaction coefficient. The components, temperature, pressure, and composition are all from Example 1.
"""

# ╔═╡ e2525963-2931-4bd4-ba32-207dc3f2413f
@htl("""
		<table>
		<caption> State specification </caption>
		  <tr>
		    <th>Components</td>
		    <td>Methane</td>
			<td>Hydrogen Sulfide</td>
		  </tr>
		  <tr>
		    <th>Mole fraction</td>
		    <td>0.5</td>
			<td>0.5</td>
		  </tr>
		  <tr>
		    <th>Temperature</td>
		    <td colspan="2">187.0 K</td>
		  </tr>
		  <tr>
		    <th>Pressure</td>
		    <td colspan="2">4.052 MPa</td>
		  </tr>
		</table>
		""")

# ╔═╡ 9711a0ba-d026-4e1a-a3f7-ba35489bc9e8
begin
	model = EPPR78(["methane", "hydrogen sulfide"])
	T = 187.0
	p = 4.052e6
	
	z = [0.5, 0.5]
end

# ╔═╡ f1ae7174-07ce-4316-a77a-32d330f0c278
md"""
### 2. Generate initial guesses

$$\ln K_i = \ln \frac{P_{c,i}}{P_i} + 5.373(1+\omega_i)\left(1-\frac{T_{c,i}}{T}\right)$$

The Wilson approximation is based on the ideal solution approximation, and is structured to match pure component vapour pressure at $T_r = 0.7$ and $T_r = 1.0$. It relies on the **critical temperature and pressure** as well as the **acentric factor**, all easily obtainable properties of the pure components. While it generally performs quite well, especially for mixtures relevant to the petrochemical industry, it has very poor predictions when used with hydrogen.

To call calculate Wilson K-factors we use the function
```julia
wilson_k_values(model, p, T)
```
"""

# ╔═╡ 2086f965-265c-49df-8971-fae0657f8d87
md"""
### 3. Minimise our objective function

Next we minimise our objective function, making sure we tell our solver to use automatic differentiation for the derivatives.
"""

# ╔═╡ ade5b56b-de17-4950-9f92-55c8c134196c
function chemical_stability_analysis(model, p, T, z)
	Kʷ = wilson_k_values(model, p, T)
	z = z ./ sum(z)
	# Generate initial guesses
	w_vap = normalize(z ./ Kʷ, 1) # vapour-like
	w_liq = normalize(Kʷ .* z, 1) # liquid-like
	w0_vec = [w_liq, w_vap]

	# Objective function - Unconstrained formulation in mole numbers
	φ(x) = fugacity_coefficient(model, p, T, x)
	d(x) = log.(x) .+ log.(φ(x))
	d_z = d(z)

	tm(W) = 1.0 + sum(W .* (d(W) .- d_z .- 1.0))
	f(W, _) = tm(exp.(W))

	# Solve for our liquid and vapour guesses
	optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
	prob(w0) = OptimizationProblem(optf, log.(w0))
	sol(w0) = solve(prob(w0), Newton())
	sol_vec = sol.(w0_vec)

	# Extract minimum
	tm_min, idx = findmin([s.minimum for s in sol_vec])
	# W = exp.(sol_vec[idx].minimizer)
	tm_xmin = normalize(exp.(sol_vec[idx].u), 1)

	# Evaluate tpd
	tpd(x) = sum(x .* (log.(x) + log.(φ(x)) .- d_z))
	tpd_min = tpd(tm_xmin)
	
	return tm_xmin, tpd_min
end

# ╔═╡ 5accf1cf-4475-4ddd-862a-b30b105462d9
tm_xmin, tpd_min = chemical_stability_analysis(model, p, T, z)

# ╔═╡ 601fb988-665a-4bb4-a412-eaa995b932f1
tp_flash(model, p, T, z)

# ╔═╡ 9f8aa417-5887-4294-b497-3925f9cd7ded
@htl("""
We can see that our value of tpd_min, $(round(tpd_min;sigdigits=4)), is less than 0. This correctly suggests that our mixture is unstable and a phase split will occur.

On top of this, if we plot the minimum point we see it's incredibly close to the actual value of the equilibrium values - this secondary function of providing initial guesses to the flash solver is part of what makes stability analysis via the Gibbs tangent plane so powerful.
""")

# ╔═╡ ebeff010-8e3e-4ab6-8215-d4b9d80288d2
md"""
In certain cases, this can provide incredibly good initial guesses. This is, unfortunately, not the norm.
"""

# ╔═╡ c8cdbae6-118d-4636-8205-47fde2a72e01
md"""
Otherwise, we often obtain pretty good initial guesses, though not perfect.
"""

# ╔═╡ 0b018495-b1ad-46b0-af2a-dde532cf3442
md"""
## Problems and difficulties

However, our stability analysis algorithm is not foolproof. If we begin to increase the temperature from our state above, we approach the critical point of Methane and a third phase begins to appear. You can see this below using our Newton algorithm we converge to the incorrect minima.

We now have two minima very close to one another, and converging to the "correct" point can become very difficult without using more expensive optimisation techniques. 
"""

# ╔═╡ 217bbc70-3db5-4368-8819-0fabb9071637
begin
	T2 = 189.7
	tm_xmin2, tm_min2 = chemical_stability_analysis(model, p, T2, z)
end

# ╔═╡ bfc93f97-258b-482f-bd4f-3781e7fd548f
md"""
## Footnotes
[^1]: 
	The original paper detailing this is:
"""

# ╔═╡ 30692b0e-6029-40a2-98a5-d8803304ff7e
DOI("10.1016/0378-3812(82)85001-2")

# ╔═╡ 5ed26e66-7c4b-466a-b95d-90c283aac6c6
md"## Function library

Just some helper functions used in the notebook."

# ╔═╡ b0429aaf-cdf4-487f-972e-8761f77cc0ad
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ b4939d68-71e4-480a-9c07-f1a6db51bb8b
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ dc6f6cb6-90df-453f-ba95-9542d81908ea
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ b05f3f59-434d-4c94-9393-d4683e27410e
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

# ╔═╡ 888a61a9-9b5e-41d6-bba5-ce259ecca3f5
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

# ╔═╡ b90ae8a7-5746-47a4-b182-90f2e8ad19c5
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ ef150bc3-65a1-4695-a1dc-ef8a2e602a5a
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 64259d11-1310-49b6-aaec-bfe9832847fa
# CHECKING FUNCTION
if !@isdefined(model)
    not_defined(:model)
else
    let
        if !(model isa EoSModel)
            still_missing(md"Make sure to define `model` using Clapeyron")

        elseif (model isa RK) && (issetequal(model.components, ["methane", "ethane"])) && (model.alpha isa SoaveAlpha)
            correct()

        else
            md"""
            !!! warning "Incorrect"
            	Make sure to define `model` using `SRK`
            """
        end
    end
end;

# ╔═╡ 86e10ead-c665-4cb1-9956-b700968bee7f
tangent_line(f, x₀) = (x -> f(x₀) + ForwardDiff.derivative(f, x₀) * (x - x₀))

# ╔═╡ 58dabd1a-c5ca-49f5-9af1-60e86838c93a
let
    model = PSRK(["octane", "ethane"])

    p = 2e5
    T = 405.0
    z = (x -> [x, 1-x])(0.8)

    gⱽ(z) = gibbs_free_energy(model, p, T, z; phase=:vapour) ./ (R * T)
    gᴸ(z) = gibbs_free_energy(model, p, T, z; phase=:liquid) ./ (R * T)
    gᴹ(z) = mixing(model, p, T, z, gibbs_free_energy) ./ (R * T)
	
	# x_out = multiphase_flash(model, p, T, z)
	x_out, _, _ = tp_flash(model, p, T, z, RRTPFlash())
    flash_xmin = x_out[1, :]
    flash_ymin = x_out[2, :]

    # Returns function describing our tangent plane
    tangent_plane(w) = tangent_line(x -> gᴹ([x, 1 - x]), w[1])

    flash_tangent = tangent_plane(flash_xmin)

	w2 = [0.08, 0.92]
	stability_tangent1 = tangent_plane([0.115, 0.885])
	stability_tangent2 = tangent_plane(w2)

	x_range = range(1e-3, 1 - 1e-3, 200)
    z_range = [[x, 1 - x] for x in x_range]

	gr()

	fig9 = plot(title="Gibbs free energy of mixing in a\noctane + ethane mixture\n",
        ylabel=L"Δg^{\mathrm{mix}}/RT", xlabel=L"x~(octane)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, xlim=(0.0, 1.0), ylim = (-0.5, 0.30),
		legend=:topleft
    )

    plot!(x_range, gᴹ.(z_range), label="Δg surface", linewidth=2)
    annotate!(0.77, 0.23, text("T=$(T) K\np=$(p/1e6) MPa", :black, :left, 14))
	
    plot!([flash_xmin[1], flash_ymin[1]], [gᴹ(flash_xmin), gᴹ(flash_ymin)], label="equilibrium tangent plane", linewidth=3, linestyle=:dash)

	tm_xmin, tpd_min = chemical_stability_analysis(model, p, T, z)
	scatter!([tm_xmin[1]], [gᴹ(tm_xmin)], markersize=4, markercolour=:black, label="tm minimum")
	fig9
end

# ╔═╡ 51f6e633-e954-4542-badc-5ddb178509d1
let # Fig 7
	gr()
    model = EPPR78(["methane", "hydrogen sulfide"])

    p = 4.052e6
    T = 190.5
    z = ones(2)/2

    gⱽ(z) = gibbs_free_energy(model, p, T, z; phase=:vapour) ./ (R * T)
    gᴸ(z) = gibbs_free_energy(model, p, T, z; phase=:liquid) ./ (R * T)
    gᴹ(z) = mixing(model, p, T, z, gibbs_free_energy) ./ (R * T)
	
    x_out, nvals, best_f = tp_flash(model, p, T, z, DETPFlash(time_limit=5))
	# x_out = multiphase_flash(model, p, T, z)
    flash_xmin = x_out[1, :]
    flash_ymin = x_out[2, :]

    # Returns function describing our tangent plane
    tangent_plane(w) = tangent_line(x -> gᴹ([x, 1 - x]), w[1])

    flash_tangent = tangent_plane(flash_xmin)
	
    global fig7 = plot(title="Gibbs free energy of mixing in a\nCH₄ + H₂S mixture\n",
        ylabel=L"Δg^{\mathrm{mix}}/RT", xlabel=L"x, y~(\mathrm{CH}_4)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, ylim = (-0.08, 0.10), xlim=(0.0, 1.0)
    )

    x_range = range(1e-3, 1 - 1e-3, 200)
    z_range = [[x, 1 - x] for x in x_range]
    plot!(x_range, gᴹ.(z_range), label="Δg surface", linewidth=2)
    annotate!(0.71, 0.08, text("T=$(T) K\np=$(p/1e6) MPa", :black, :left, 14))

    flash_range = range(flash_xmin[1], flash_ymin[1], 3)
    plot!([flash_xmin[1], flash_ymin[1]], [gᴹ(flash_xmin), gᴹ(flash_ymin)], label="equilibrium tangent plane", linewidth=3, linestyle=:dash)

	scatter!([tm_xmin2[1]], [gᴹ(tm_xmin2)], markersize=4, markercolour=:black, label="tm minimum")
	# stability_range = [0.01, 0.99]
    plot!(legend=:topleft)
end;

# ╔═╡ 3dafafd0-99ea-4f36-82ae-4a28d090c866
fig7

# ╔═╡ a0b8bd11-282c-4bc6-9458-668bd9b007eb
# Fig 1, 5, 6, 8
let
    model = EPPR78(["methane", "hydrogen sulfide"])
	
    p = 4.052e6
    T = 187.0
    z = (x -> [x, 1-x])(0.5)

    gⱽ(z) = gibbs_free_energy(model, p, T, z; phase=:vapour) ./ (R * T)
    gᴸ(z) = gibbs_free_energy(model, p, T, z; phase=:liquid) ./ (R * T)
    gᴹ(z) = mixing(model, p, T, z, gibbs_free_energy) ./ (R * T)

	# Fig 1 - 2D tangent plane criteria
	
	# x_out = multiphase_flash(model, p, T, z)
	x_out, _, _ = tp_flash(model, p, T, z, DETPFlash(time_limit=5))
    flash_xmin = x_out[1, :]
    flash_ymin = x_out[2, :]

    # Returns function describing our tangent plane
    tangent_plane(w) = tangent_line(x -> gᴹ([x, 1 - x]), w[1])

    flash_tangent = tangent_plane(flash_xmin)

	w2 = [0.08, 0.92]
	stability_tangent1 = tangent_plane([0.115, 0.885])
	stability_tangent2 = tangent_plane(w2)

	gr()
    global fig1 = plot(title="Gibbs free energy of mixing in a\nCH₄ + H₂S mixture\n",
        ylabel=L"Δg^{\mathrm{mix}}/RT", xlabel=L"x~(\mathrm{CH}_4)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, ylim = (-0.08, 0.10), xlim=(0.0, 1.0),
		legend=:topleft
    )

    x_range = range(1e-3, 1 - 1e-3, 200)
    z_range = [[x, 1 - x] for x in x_range]
	stability_range = [0.01, 0.99]
    flash_range = range(flash_xmin[1], flash_ymin[1], 3)
	
    plot!(x_range, gᴹ.(z_range), label="Δg surface", linewidth=2)
    annotate!(0.71, 0.08, text("T=$(T) K\np=$(p/1e6) MPa", :black, :left, 14))

    plot!([flash_xmin[1], flash_ymin[1]], [gᴹ(flash_xmin), gᴹ(flash_ymin)], label="equilibrium tangent plane", linewidth=3, linestyle=:dash)

	plot!(x_range, stability_tangent1.(x_range), label="test tangent plane 1", linewidth=3, linestyle=:dash)
	plot!(x_range, stability_tangent2.(x_range), label="test tangent plane 2", linewidth=3, linestyle=:dash)
	
    global fig5 = plot(title="Gibbs free energy of mixing in a\nCH₄ + H₂S mixture\n",
        ylabel=L"Δg^{\mathrm{mix}}/RT", xlabel=L"x~(\mathrm{CH}_4)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, ylim = (-0.08, 0.10), xlim=(0.0, 1.0),
		legend=:topleft
    )

    plot!(x_range, gᴹ.(z_range), label="Δg surface", linewidth=2)
    annotate!(0.71, 0.08, text("T=$(T) K\np=$(p/1e6) MPa", :black, :left, 14))

	plot!(x_range, stability_tangent2.(x_range), label="test tangent plane", linewidth=3, linestyle=:dash)
	plot!([z[1],z[1]],[gᴹ(z),stability_tangent2(z[1])], arrow=true,color=:black,linewidth=2,label="")
	annotate!([z[1]-0.02], [mean([gᴹ(z),stability_tangent2(z[1])])], text("tangent\nplane\ndistance", :black, :right, 12))
	annotate!(w2[1], stability_tangent2(w2[1])-0.01, text("x", :black, :centre, 12))
	annotate!(z[1], gᴹ(z)+0.01, text("z", :black, :centre, 12))

	global fig6 = plot(title="Gibbs free energy of mixing in a\nCH₄ + H₂S mixture\n",
        ylabel=L"Δg^{\mathrm{mix}}/RT", xlabel=L"x~(\mathrm{CH}_4)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, ylim = (-0.08, 0.10), xlim=(0.0, 1.0),
		legend=:topleft
    )

    plot!(x_range, gᴹ.(z_range), label="Δg surface", linewidth=2)
    annotate!(0.71, 0.08, text("T=$(T) K\np=$(p/1e6) MPa", :black, :left, 14))
	
    plot!([flash_xmin[1], flash_ymin[1]], [gᴹ(flash_xmin), gᴹ(flash_ymin)], label="equilibrium tangent plane", linewidth=3, linestyle=:dash)

	scatter!([tm_xmin[1]], [gᴹ(tm_xmin)], markersize=4, markercolour=:black, label="tm minimum")
	
	global fig8 = plot(title="Tangent plane distance for a CH₄ + H₂S mixture",
        ylabel=L"TPD", xlabel=L"x~(\mathrm{CH}_4)", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off, xlim=(0.0, 1.0), legend=:topleft
    )
	
	μ(x) = chemical_potential(model, p, T, x)
	TPD(x) = sum(x.*(μ(x) .- μ(z)))
	plot!(x_range, TPD.(z_range), label="", linewidth=2)
end;

# ╔═╡ ee9a3700-848e-4827-bb26-81a555b01a33
fig1

# ╔═╡ 8502b61a-c88a-4c7e-9237-1d6315f88534
fig5

# ╔═╡ 798167c9-79d1-43b2-8902-c372c54a6916
fig8;

# ╔═╡ 9890eb1e-cc4e-4d9a-a47e-348c624a2934
fig6

# ╔═╡ 06e8ef01-5d9f-4683-a2a6-46f531009253
# Fig 2, 3
let
    # model = EPPR78(["hydrogen sulfide", "carbon monoxide", "methane"])
    # model = EPPR78(["nonane", "carbon monoxide", "hydrogen sulfide"])
    model = PSRK(["octane", "ethane", "propane"])
    # model = PR(["ethane", "propane", "octane"])
	
    p = 1e6
    T = 300.0
	
    gᴹ(z) = mixing(model, p, T, z, gibbs_free_energy) ./ (R * T)
	
	# Fig 2 - 3D Gibbs surface plot
	function gᴹ_plot(x, y) 
		if x + y ≤ 1.0-1e-4
			return mixing(model, p, T, [x, y, 1.0-x-y], gibbs_free_energy) ./ (R * T)
		else
			return NaN
		end
	end

    x_range = range(1e-4, 1.0-1e-4, 200)
	y_range = range(1e-4, 1.0-1e-4, 200)
	
	g_plot = [gᴹ_plot(x, y) for x in x_range, y in y_range]

	plotlyjs()
	
	global fig2 = plot(title="Gibbs free energy of mixing", xlabel="x (octane)", ylabel="x (ethane)", zlabel="Δg/RT", yguidefontsize=16, xguidefontsize=16, legendfont=font(10), framestyle=:box, tick_direction=:out, grid=:off)
	# global outvar = g_plot
	surface!(x_range, y_range, g_plot) #, label = "Δg surface"

	# x_ternary = []
	# y_ternary = []
	# z_ternary = []
	# # g_ternary = reshape(g_plot, :, 1)
	# g_ternary = []
	# for x in x_range
	# 	for y in y_range
	# 		if x + y < 1.0-1e-4
	# 			z = 1.0 - x - y
	# 			push!(x_ternary, x)
	# 			push!(y_ternary, y)
	# 			push!(z_ternary, z)
	# 			push!(g_ternary, gᴹ_plot(x, y))
	# 		end
	# 	end
	# end
	# g_ternary = [gᴹ_plot(x, y) for x in x_range, y in y_range]
	
	# largest = maximum(abs.([minimum(g_ternary), maximum(g_ternary)]))
	# Cmap_β = GMT.makecpt(cmap=:rainbow, T=(0, 1))
	# C_new = GMT.makecpt(cmap=:panoply, T=(-largest, largest))
	# C_new.bfn[3,:] = [1. 1 1];       # Force the NaNs to be painted as white

	# GMT.ternary(
	# 	cmap=C_new, MarkerSize=0,
	# 	hcat(x_ternary, y_ternary, z_ternary, g_ternary),
	# 	# clockwise=true,
	# 	image=true,
	# 	contour=(annot=0.1, cont=0.1),
	#     frame=(annot=:auto, grid=:a, ticks=:a, alabel="octane", blabel="ethane", clabel="propane", suffix=" %")
	# )

	# Calculate equilibrium compositions
	# z_flash = [0.1, 0.1, 0.8]
	# x_envelope = []
	# for z in z_plot
	# 	x, n, G = tp_flash(model, p, T, z_flash, RRTPFlash())
	# end

	# for z_flash in 
	z_flash = [[0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]]
	# z_flash = [[i, 1-i-1e-1, 1e-1] for i in range(1e-1, 0.95, 100)]
	# for z in z_flash
	# 	try
	# 		stable, xmin = check_chemical_stability(model, p, T, z)
	# 		if ~stable
	# 			if length(xmin) == 2
	# 				K0 = xmin[1]./xmin[2]
	# 			else
	# 				K0 = Clapeyron.wilson_k_values(model, p, T, z)
	# 			end
	# 			x_flash, n, G = tp_flash(model, p, T, z, RRTPFlash(;K0=K0))
	# 			# x_flash, n, G = tp_flash(model, p, T, z, DETPFlash(time_limit=1))
				
	# 			# GMT.plot!(GMT.tern2cart(x_flash), marker=p, MarkerSize=0.3, fill=:red)
	# 			# GMT.plot!(GMT.tern2cart(x_flash), marker=p, lw=2, lc=:red)
	# 			# GMT.plot!(GMT.tern2cart([z[1] z[2] z[3]]), marker=p, MarkerSize=0.3, fill=:green)
	# 		end
	# 	catch e
	# 	end
	# end

	# Manual flash points
	# flash_values = readdlm("output.txt", Float64)
	# flash_values = flash_values[0.0 .< flash_values[:,4] .< 1e-2, 1:3]
	# GMT.plot!(GMT.tern2cart(flash_values), marker=p, MarkerSize=0.1, fill=:red)
	
	# Manual legend - This is kinda poor lol
	# GMT.text!([0.09 0.85], text="T = $T K", font=14)
	# GMT.text!([0.1 0.8], text="p = $(p/1e6)  MPa", font=14)

	# GMT.plot!([0.01 0.73], marker=p, MarkerSize=0.3, fill=:red)
	# GMT.text!([0.13 0.745], text="equilibrium", font=12)
	# GMT.text!([0.13 0.715], text="compositions", font=12)
	
	# GMT.plot!([0.01 0.66], marker=p, MarkerSize=0.3, fill=:green)
	# GMT.text!([0.13 0.66+0.015], text="overall", font=12)
	# GMT.text!([0.13 0.66-0.015], text="composition", font=12)

	# global fig3 = GMT.colorbar!(xlabel=raw"delta g_mix / RT", pos=(outside=:RR,), frame=:auto, show=true)
end;

# ╔═╡ 7782e58c-fed7-4920-81dd-02d7800ce8e6
fig2

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clapeyron = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Clapeyron = "~0.3.7"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
NLsolve = "~4.5.1"
Optimization = "~3.7.1"
OptimizationOptimJL = "~0.1.1"
PlotlyJS = "~0.18.8"
Plots = "~1.31.2"
PlutoUI = "~0.7.39"
ShortCodes = "~0.3.3"
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
git-tree-sha1 = "7d255eb1d2e409335835dc8624c35d97453011eb"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.14"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

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
git-tree-sha1 = "2dd813e5f2f7eec2d1268c57cf2373d3ee91fcea"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.1"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

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
git-tree-sha1 = "a599cfb8b1909b0f97c5e1b923ab92e1c0406076"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.1"

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
git-tree-sha1 = "429077fd74119f5ac495857fd51f4120baf36355"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.65"

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
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "e3af8444c9916abed11f4357c2f59b6801e5b376"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.13.1"

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
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "4078d3557ab15dd9fe6a0cf6f65e3d4937e98427"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.0"

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

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

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
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

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
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

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
git-tree-sha1 = "891d3b4e8f8415f53108b4918d0183e61e18015b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.0"

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

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "bfbd6fb946d967794498790aa7a0e6cdf1120f41"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.13"

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
git-tree-sha1 = "cf1f5812820ddcb8efa2d1a5eb582aa3a93058d6"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "3.7.1"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "Optimization", "Reexport", "SparseArrays"]
git-tree-sha1 = "9f3a70a410b659abe9187b348cf357d1dbaf4257"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.1.1"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "b29873144e57f9fcf8d41d107138a4378e035298"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.2"

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
git-tree-sha1 = "7a5f08bdeb79cf3f8ce60125fe1b2a04041c1d26"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.31.1"

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
git-tree-sha1 = "30e3981751855e2340e9b524ab58c1ec85c36f33"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "55f38a183d472deb6893bdc3a962a13ea10c60e4"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.42.4"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "e972716025466461a3dc1588d9168334b71aafff"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.1"

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
git-tree-sha1 = "48598584bacbebf7d30e20880438ed1d24b7c7d6"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.18"

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
git-tree-sha1 = "79aa7175f0149ba2fe22b96a271f4024429de02d"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.9.0"

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

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

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
# ╠═a598b2d2-f15c-11ec-10ed-a15ca6a73a60
# ╟─04089557-9381-4bbb-9882-45b5a685694b
# ╠═ee9a3700-848e-4827-bb26-81a555b01a33
# ╟─cc88e502-8b06-44fc-b888-3bed9cf1b683
# ╟─87bb11a1-d8f7-4b51-b67a-2cfeef4ad039
# ╟─7782e58c-fed7-4920-81dd-02d7800ce8e6
# ╟─f1c2c57d-0d05-48d4-ae6d-a015001d1888
# ╟─8502b61a-c88a-4c7e-9237-1d6315f88534
# ╟─798167c9-79d1-43b2-8902-c372c54a6916
# ╟─81a77642-52ef-4836-9055-5059332f72d8
# ╟─64259d11-1310-49b6-aaec-bfe9832847fa
# ╟─1db67ed3-53cf-42a9-a8a0-be3e2a2dc537
# ╟─e2525963-2931-4bd4-ba32-207dc3f2413f
# ╠═9711a0ba-d026-4e1a-a3f7-ba35489bc9e8
# ╟─f1ae7174-07ce-4316-a77a-32d330f0c278
# ╟─2086f965-265c-49df-8971-fae0657f8d87
# ╠═ade5b56b-de17-4950-9f92-55c8c134196c
# ╠═5accf1cf-4475-4ddd-862a-b30b105462d9
# ╠═601fb988-665a-4bb4-a412-eaa995b932f1
# ╟─9f8aa417-5887-4294-b497-3925f9cd7ded
# ╟─ebeff010-8e3e-4ab6-8215-d4b9d80288d2
# ╟─9890eb1e-cc4e-4d9a-a47e-348c624a2934
# ╟─c8cdbae6-118d-4636-8205-47fde2a72e01
# ╟─58dabd1a-c5ca-49f5-9af1-60e86838c93a
# ╟─0b018495-b1ad-46b0-af2a-dde532cf3442
# ╟─51f6e633-e954-4542-badc-5ddb178509d1
# ╟─217bbc70-3db5-4368-8819-0fabb9071637
# ╟─3dafafd0-99ea-4f36-82ae-4a28d090c866
# ╟─bfc93f97-258b-482f-bd4f-3781e7fd548f
# ╟─30692b0e-6029-40a2-98a5-d8803304ff7e
# ╟─5ed26e66-7c4b-466a-b95d-90c283aac6c6
# ╟─b0429aaf-cdf4-487f-972e-8761f77cc0ad
# ╟─b4939d68-71e4-480a-9c07-f1a6db51bb8b
# ╟─dc6f6cb6-90df-453f-ba95-9542d81908ea
# ╟─b05f3f59-434d-4c94-9393-d4683e27410e
# ╟─888a61a9-9b5e-41d6-bba5-ce259ecca3f5
# ╟─b90ae8a7-5746-47a4-b182-90f2e8ad19c5
# ╟─ef150bc3-65a1-4695-a1dc-ef8a2e602a5a
# ╟─86e10ead-c665-4cb1-9956-b700968bee7f
# ╟─a0b8bd11-282c-4bc6-9458-668bd9b007eb
# ╟─06e8ef01-5d9f-4683-a2a6-46f531009253
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
