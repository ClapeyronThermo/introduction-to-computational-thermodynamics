### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 061fb273-ec2a-4e89-ba73-c4d3c4a76176
begin
	using Clapeyron, ForwardDiff, Roots, Optim, LinearAlgebra, PolynomialRoots # Numerical packages
	using LaTeXStrings, Plots, ShortCodes, Printf # Display and plotting
	using HypertextLiteral, PlutoUI
	# using JSON2, Tables,Random # Data handling
	using BenchmarkTools
	PlutoUI.TableOfContents()
end

# ╔═╡ f9c5a11e-0e45-11ed-2b32-153050d16a4a
md"""
# Section 2.4 - Statistical Association Fluid Theory
_N.B.: This will be a high-level overview of the SAFT equations, with more emphasis on the physical picture than the implementation. If you were able to implement the generalised cubic equation in the previous section, you should be able to implement any of the SAFT equations (with the exception of one aspect which will be discussed here)._

We previously established that, while the cubics are some of the most-flexible equations of state, the range of systems and properties you can model accurately with them is limited. Taking a few steps back, we mentioned previously that the van der Waals equation can actually be derived analytically. Writing out the residual Helmholtz free energy:

$$A_\mathrm{res.} = -nRT\log{(1-nb/V)}-\frac{n^2a}{V}$$

we can imagine that the first term represents the contribution from the presence of particles as hard spheres ($A_\mathrm{HS}$) and the second is a perturbation from those hard spheres to account for dispersive, pair-wise interactions ($A_1$). Thus, we could write it out as:

$$A_\mathrm{res.} = A_\mathrm{HS}+A_1$$

Note that $A_\mathrm{HS}$ has been refined in literature to better model true hard-sphere systems. Nevertheless, one way to improve upon this equation is to take higher-order perturbations from the hard-sphere model to account for many-body interactions:

$$A_\mathrm{res.} = A_\mathrm{HS}+A_1+\frac{A_2}{Nk_\mathrm{B}T}+...$$

In such approaches, species are no longer characterised by the parameters $a$ and $b$, we now use the diameter of our species ($\sigma$) and the dispersive energy parameter ($\epsilon$). Visually:

"""

# ╔═╡ 9496586f-63f2-4cb8-9ece-7d571dd68d4e
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Seg.svg" height="300"></center>""")

# ╔═╡ 5e449695-f7b2-452c-82b2-af193e776280
md"""
Despite these improvements, it doesn't change the fact that we are just modelling spherical systems with dispersion interactions, albeit more accurately; most species will not fit this description. For decades after van der Waals' derived his equation, researchers have developed new approaches to more accurately model a larger range of species. In 1989, Chapman _et al._ published the Statistical Associating Fluid Theory (SAFT) where they first grouped the hard-sphere and perturbation contributions into a single term (the segment term, $A_\mathrm{seg.}$) and introduced two new terms:

$$A_\mathrm{res.} = A_\mathrm{seg.}+A_\mathrm{chain}+A_\mathrm{assoc.}$$
### Chain term
The first new term is the chain term. Here, we account for the formation of a chain made up of $m$ segments (thus introducing one new parameter). This term allows us to better model chain-like species (anything from alkanes to polymers):
"""

# ╔═╡ ae398071-8969-4b6d-ba6c-12122c2c5b94
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Chain.svg" height="400"></center>""")

# ╔═╡ 762c0786-9152-429f-906f-728056200a9a
md"""
One thing to bear in mind is that this new $m$ parameter, the number of segments doesn't necessarily represent the number of monomers (e.g., number of CH$_2$ groups in alkanes) and doesn't even have to be an integer (in this case, a non-integer value can be imagined as a 'merging' of spheres). This $m$ is just a 'best-fit' for the number of hard spheres to represent a molecule.

### Association term
The second term introduced by Chapman _et al._ is the association term. Here, species are modelled as having small, tangential sites which can overlap with the sites from other species to form a dimer:
"""

# ╔═╡ 2609eb20-7f6c-455b-a1b3-f4cb99d1e50c
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Assoc.svg" height="400"></center>""")

# ╔═╡ 25437516-b88b-478c-b6ab-be0a293305de
md"""
The idea here is that this approach can be used to mimic hydrogen bonding in systems like water. This interaction is characterised by an association energy parameter ($\epsilon^\mathrm{assoc.}_{ij,ab}$) and a bonding 'volume' (either dimensionless, $\kappa_{ij,ab}$ or non-dimensionless, $K_{ij,ab}$).

Whilst the segment and chain terms are tedious, but explicit equations which simply take a lot of time to code, the association term introduces one additional level of complexity. At the core of the association term is the association fraction of a site $a$ on species $i$, $X_{i,a}$, which is given by:

$$X_{ia} = \frac{1}{1 +\sum_{j,b}{\rho_{j}\Delta_{ijab}X_{jb}}}\,,$$

where $\Delta_{ij,ab}$ is the association strength between site $a$ on species $i$ and site $b$ on species $j$. This equation, at first, appears a bit daunting. If we simplify it so that we only have one species with one site which can only bond with itself, we have:

$$X = \frac{1}{1 +\rho \Delta X}\,.$$

The issue should now become more apparent: this equation is implicit (i.e. $y=f(x,y)$). This means we will need to use an iterative method to solve for the association fraction $X$. Thankfully, in the above case, the solution can actually be solved for explicitly as:

$$X = \frac{-1+\sqrt{1+4\rho\Delta}}{2\rho\Delta}$$

However, this may not always be the case; for most mixtures with multiple sites, no analytical solution exists. One option is to use successive substitution, where we simply have $X^{(i+1)}=f(X^{(i)})$. In code:
"""

# ╔═╡ 0b3545ea-ab28-4de3-9be2-91d3a6cccea8
begin
	X = [1.]
	tol = 1.
	ρΔ = 10.
	i = 1
	while tol>1e-6
		X_new = (1+ρΔ*X[i])^-1
		append!(X,X_new)
		tol = abs(X[i+1]-X[i])
		i+=1
	end
end

# ╔═╡ fe094bfb-65a7-4549-9db9-d27ece7c79a5
md"""
The issue with this approach is that it oscillates around the true solution:
"""

# ╔═╡ 704ed6c5-6730-42a9-bdbc-6b04e4b666b7
begin
	X_true = (-1+sqrt(1+4*ρΔ))/(2*ρΔ)

	plot(X_true*ones(length(X)),xlim=(1,10),ylim=(0,1),
		title="Successive substitution for association fraction",
		label="Exact",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"X", xlabel=L"i")
	plot!(X,label="Successive Substitution")

end

# ╔═╡ 02f3060e-df47-46a3-b1fa-37f10959b86a
md"""
One easy solution is to introduce a damping factor where, rather than accepting the new solution, we 'damp it' using some of our current solution:

$$X^{(i+1)}=\alpha X^{(i)}+(1-\alpha)f(X^{(i)})$$

After much experimentation in literature, the optimal damping factor, $\alpha$, is found to be 0.5. This is enough to eliminate the oscillations but still converge relatively quickly:
"""

# ╔═╡ 49a74e5c-c17b-40a8-8bc4-f62ae3131fc3
α = 0.5;

# ╔═╡ c76a488b-d93c-403c-8ebf-44ca79bbd849
begin
	X1 = [1.]
	tol1 = 1.
	j = 1
	while tol1>1e-6
		X_new = α*X1[j]+(1-α)*(1+ρΔ*X1[j])^-1
		append!(X1,X_new)
		tol1 = abs(X1[j+1]-X1[j])
		j+=1
	end
	plot(X_true*ones(length(X)),xlim=(1,10),ylim=(0,1),
		title="Successive substitution for association fraction",
		label="Exact",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"X", xlabel=L"i")
	plot!(X,label="Successive Substitution")
	plot!(X1,label="Damped Successive Substitution")
end
	

# ╔═╡ 1aed05c7-68e9-466c-a449-3ad6c3fdebb9
md"""
Naturally, wherever possible, it is best to use the analytical solution and only use the iterative method when all else fails. There have been other methods developed in literature which speed up convergence using Newton's method but this is beyond the scope of this course. Nevertheless, with the tools at hand, you should be able to implement your own SAFT equation.
"""

# ╔═╡ d2dedb47-6d78-40a7-8b47-85d86ae2fa7a
md"""
### Mixtures
The SAFT equations are derived based on pure-component systems. As such, like the cubics, SAFT equations need to use mixing rules. Whereas with cubics, we applied mixing rules to the parameters (which we do with some SAFT equations), for most SAFT equations we apply mixing rules on the Helmholtz free energy directly. This is handled on a term-by-term basis. Firstly, in the segment term, a quadratic mixing rule is used:

$$\bar{A}_\mathrm{seg.}=\sum_i\sum_jx_ix_j A_{\mathrm{seg.},ij}$$

where, for $i=j$, $A_{\mathrm{seg.},ii}$ is just the segment contribution for a pure component $i$. For $i\neq j$, $A_{\mathrm{seg.},ij}$ now represents the contribution from the fictitious fluid which is characterised by parameters:

$$\sigma_{ij}=\frac{\sigma_{ii}+\sigma_{jj}}{2}(1-l_{ij})$$
$$\epsilon_{ij}=\sqrt{\epsilon_{ii}\epsilon_{jj}}(1-k_{ij})$$

Again, we have our 'fudge factors' $l_{ij}$ and $k_{ij}$, which, like the cubics, must be fit using experimental data.

For the chain term, a linear mixing rule is used instead, which requires no new parameters:

$$\bar{A}_\mathrm{chain}=\sum_ix_i A_{\mathrm{chain},i}$$

The association term is far more-complex as we can have association between sites on different species (e.g. hydrogen bonding between water and methanol). These are all accounted for in the full equation for the association fraction above. 
"""

# ╔═╡ 9a2ed4e6-a39a-4e92-b075-cfc6b62b3795
md"""
### Summary
Just to wrap this section up, the physical picture in SAFT equations is, effectively, a system made up of attracting hard spheres in a chain with association sites to mimic hydrogen bonding. Visually:
"""

# ╔═╡ da387b61-a0f2-496f-be79-eccf3d9394cc
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/SAFT.svg" height="500"></center>""")

# ╔═╡ c1a79228-e556-4d6e-9f1d-d88efc1747ed
md"""
As one can imagine, in contrast to the van der Waals equation, the range of molecules which can be modelled with SAFT equations is far more extensive. The only difficulty is the parameters which, to recap, are:
* The size of one segment in a species: $\sigma_{ii}$
* The dispersive energy of a species: $\epsilon_{ii}$
* The number of segments in a chain of a species: $m_i$
* Binary interaction parameters between species: $k_{ij}$ and $l_{ij}$
* The association energy between sites on species: $\epsilon^\mathrm{assoc.}_{ij,ab}$
* The 'bonding' volume: $\kappa^\mathrm{assoc.}_{ij,ab}$ / $K^\mathrm{assoc.}_{ij,ab}$
Note that the last two parameters are only needed for species which experience association. Unfortunately, while the parameters for the cubics can easily be obtained using information from the critical point and acentricity, this is not possible for SAFT equations. The pure parameters are usually obtained by regressing them using experimental data (usually the saturation pressure and saturated liquid densities). Thus, despite its strong physical foundation, SAFT equations are still limited by available experimental data (there are exceptions to this which will be shown latter). Thankfully, large databases of parameters are available online which should allow one to model most species of interest.

Before going online to search for parameters, there is one key detail to keep in mind: there are many different variants of SAFT equations. Whilst the formalism described above is generally true, some minor differences will be described below. We will focus on the more-popular and commonly-used SAFT equations.
"""

# ╔═╡ 303fba91-9844-4649-82be-e169d6f52a9c
md"""
## Section 2.4.1 - Perturbed-Chain SAFT (PC-SAFT)
"""

# ╔═╡ 89ff71b7-130d-4642-9558-e49efa51247e
md"""
Possibly the most-popular variant of the SAFT equation, Perturbed-Chain SAFT (PC-SAFT) was developed by Gross and Sadowski. The key difference in this equation is primarily how the terms are arranged:

$$A_\mathrm{res.} = A_\mathrm{HC}+A_\mathrm{disp.}+A_\mathrm{assoc.}$$

Here, we start with a hard-chain (HC) reference system. All this is, really, is the hard-sphere term and standard chain term summed in a single term ($A_\mathrm{HC}=A_\mathrm{HS}+A_\mathrm{chain}$). The big change comes from the fact that, rather than perturbing the hard-sphere system and then forming the chain, in PC-SAFT, we do it the other way around: the chain is formed first and then this system is perturbed to account for dispersive interactions ($A_\mathrm{disp.}$). This difference is quite subtle, but, it does result in a more-physically sound model. The association term is unchanged from the original formulation. Visually:
"""

# ╔═╡ 19f044fc-c181-44d8-bf4f-3609eade334c
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/PCSAFT.svg" height="500"></center>""")

# ╔═╡ 76959b4f-ae28-4b9a-b24e-572fccced269
md"""
The primary reasons behind PC-SAFT's popularity are three-fold. For one, the code for PC-SAFT was available open-source from publication. Secondly, there is an abundance of parameters available (over 250 species), including binary-interactions parameters. Finally, many variants of the PC-SAFT equation have been developed. This last point, unfortunately, is actually one of the downsides of PC-SAFT: one has to be very careful which version of PC-SAFT is being used. Just to name a few:
* Polar PC-SAFT (PPC-SAFT): An additional term is added to the PC-SAFT equation to account for dipole interactions. With it, we introduce a new parameter, $\mu_i$, the dipole moment of a segment. The overall equation is:
$$A_\mathrm{res.} = A_\mathrm{HC}+A_\mathrm{disp.}+A_\mathrm{assoc.}+A_\mathrm{polar}$$
* Group-contribution PC-SAFT (GC-PC-SAFT): A group-contribution method for PC-SAFT which allows users to model a larger range of species. Unfortunately, there are different ways group-contribution methods are handled in PC-SAFT, so one still needs to be consistent.
* Critical-point PC-SAFT (CP-PC-SAFT): As a way to become more like a cubic, with this PC-SAFT variant, the parameters are based on the critical point.
* Simplified PC-SAFT (sPC-SAFT): An attempt to reduce computational costs, with this version, many of the equations used in the standard PC-SAFT equation are simplified.
Furthermore, as a warning when one looks for parameters, implementations exist in which the above approaches are mixed. Generally speaking, PC-SAFT alone is most commonly used. 
"""

# ╔═╡ 909b14ac-5b9d-40d9-959b-9b6c171d1b3a
md"""
## Section 2.4.2 - Cubic plus association (CPA)
"""

# ╔═╡ cb1b6005-39c0-4247-b824-65a25b79dc33
md"""
As the name implies, the idea behind the cubic plus association (CPA) equation of state is that, most species can be modelled quite accurately using a cubic; only associating species are problematic. Thus, what if we keep the residual term from the cubic but add the association term at the end? This gives us a free energy of the form:

$$A_\mathrm{CPA} = A_\mathrm{SRK}+A_\mathrm{assoc.}$$

From an alternative point of view, we are replacing the chain and segment terms with a cubic equation of state. This simplifies things in terms of the parameters as, for non-associating species, we can still use the parameters we would use in our original cubic. For associating species, we replace the $\sigma_{ii}$, $\epsilon_{ii}$ and $m_i$ parameters with $a_{ii}$, $b_{ii}$ and $m_i$[^1], where $m_i$ is used in a new $\alpha$ function:

$$\alpha_i = (1+m_i(1-\sqrt{T/T_c}))^2$$

This equation has two key benefits: 
1. We only need to fit parameters for associating species (water, methanol, etc.).
2. All the tools we developed for cubics (new $\alpha$ functions, volume translation methods, advanced mixing rules, etc.) can be applied to CPA.

This makes CPA a very popular alternative to other SAFT equations because of its close link to cubics. Naturally, because it loses the physical foundation that SAFT equations are based on, it can't be used to model as wide a range of species and properties as accurately as SAFT equations can.
"""

# ╔═╡ 76790f3f-b101-48d8-b6b0-fc1df6e14e0f
md"""
## Section 2.4.3 - SAFT-VR Mie
"""

# ╔═╡ b13b2c8c-6366-43e9-936b-6deb67be6db1
md"""
The Variable-Range SAFT (SAFT-VR) framework was originally developed by Gil-Villegas _et al._. Previously, with most SAFT equations, it is assumed that a Lennard-Jones potential could be used to represent the dispersive interactions. In SAFT-VR, the idea was to use potentials where one could vary the shape of the potential. The simplest of these was the square-well potential where a new potential shape parameter ($\lambda$) was introduced, which measured the range of the dispersive interactions:
"""

# ╔═╡ 44256338-ce74-448a-a5e2-5bca38e3c2e0
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/SW.svg" height="500"></center>""")

# ╔═╡ c8de4d65-190b-4f36-b9ea-82bfebf9c632
md"""
However, this did not really improve the modelling of species much. It wasn't until Lafitte _et al._ extended this concept of a variable-range potential to a Mie potential that significant improvements were realised. This introduced two new potential shape parameters, $\lambda_\mathrm{a}$ (characterising the attractive part) and $\lambda_\mathrm{r}$ (characterising the repulsive part): 
"""

# ╔═╡ c0eb6b68-29b1-413f-a38e-c6781f9c4552
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Mie.svg" height="500"></center>""")

# ╔═╡ 716676d5-b6f5-4932-b5c7-b166a2f8faa3
md"""
This led to the development of the SAFT-VR Mie equation of state. At the surface, the formulation is generally the same as the standard SAFT equation:

$$A_\mathrm{res.} = A_\mathrm{seg.}+A_\mathrm{chain}+A_\mathrm{assoc.}$$

However, internally, two big changes were made:
* The segment term was taken to a higher-order perturbation. Up to this point, all SAFT equations only went up to a second-order perturbation expansion. For SAFT-VR Mie, it was decided to go up to third order. Physically, this means that many-body interactions in SAFT-VR Mie are better characterised. Practically, it improved the modelling of properties near the critical point and the modelling of bulk properties in general.
* Similar to PC-SAFT, while we do perturb the hard-sphere system first and then form a chain, for SAFT-VR Mie, in effect, a perturbation of the chain is also applied. This is a more-theoretically consistent implementation of the chain term.
These two modifications have led SAFT-VR Mie to become, possibly, the most advanced equation of state to date. Unfortunately, because of the introduction of two additional parameters ($\lambda_\mathrm{a}$ and $\lambda_\mathrm{r}$), it does mean more parameters need to be fitted (although, generally, it is assumed that $\lambda_\mathrm{a}=6$). As it is slightly newer than PC-SAFT, not as many parameters have been fitted for SAFT-VR Mie, limiting its range of applicability. However, a group-contribution method for SAFT-VR Mie was also developped, which will be discussed next.
"""

# ╔═╡ a1dad25b-453c-4689-b04c-39a51d1c90b8
md"""
## Section 2.4.4 - Heterosegmented SAFT-VR Mie (SAFT-$\gamma$ Mie)
"""

# ╔═╡ 0ba8192f-5e63-4fb7-990e-272031e99d21
md"""
To be more-widely applicable, rather than fit many parameters in SAFT-VR Mie, the developers decided to instead focus on developing a group-contribution method, known as SAFT-$\gamma$ Mie. However, another modification was made to SAFT-$\gamma$ Mie that further improved its ability to model a wider range of species. In both PC-SAFT and SAFT-VR Mie, it is assumed that the segments in a chain for a given molecule all have the same size. As the chains in SAFT-$\gamma$ Mie are assembled of groups, which can have different sizes, this limitation does not apply to SAFT-$\gamma$ Mie. This is why it can often be thought of as a heterosegmented version of SAFT-VR Mie. Visually:
"""

# ╔═╡ dc05c788-96f9-4c4f-b7bb-5e007e6fd9ee
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/SAFTMie.svg" height="500"></center>""")

# ╔═╡ 9b3e9913-7109-4cdc-901f-74191a3aa1b2
md"""
Unfortunately, despite this improved physical picture, the approach does still suffer from the limitations of a group-contribution approach. Nevertheless, with over 50 groups so far, SAFT-$\gamma$ Mie is often thought of as the state-of-the-art SAFT equation of state.
"""

# ╔═╡ d0d9ed02-b51c-4d6f-8bd2-4c89041a3b50
md"""
## Section 2.4.5 - Comparing SAFT equations
"""

# ╔═╡ 66ebf47c-aecc-42d6-b67d-8837d5086a44
md"""
Having now covered most of the SAFT equations one might encounter, one question remains: which one should you use? Unsurprisingly, the answer is: it depends. Speaking in generalities, most SAFT models, if parameterised correctly, will perform almost equivalently. For example, consider methanol below:
"""

# ╔═╡ dda0f661-d548-4022-9002-b1507bb998a1
md"""
There is certainly a substantial improvement in the predictions of liquid densities using any of the SAFT models, especially when compared to a standard cubic like Peng-Robinson (PR). However, a more-subtle detail appears when we approach the critical point: while the cubic capture exactly the critical temperature, all SAFT equations over-estimate it, due to how the parameters are fitted. This is something inherent to all perturbation-based theories. With SAFT-VR Mie, we attempt to address this by extending its perturbation to third order. Even then, the critical temperature is overestimated by a few Kelvin. Surprisingly, even a small difference in the estimation of the critical temperature can have a dramatic impact on bulk properties of our species:
"""

# ╔═╡ 7742a363-e2a0-4792-bc07-26f3650bb9d0
Exp_MeOH = [450.00 129.78;
455.00 132.56;
460.00 135.54;
465.00 138.75;
470.00 142.23;
475.00 146.05;
480.00 150.30;
485.00 155.08;
490.00 160.59;
495.00 167.09;
500.00 175.00;
505.00 185.03;
510.00 198.19;
515.00 215.78;
520.00 238.87;
525.00 271.91;
530.00 348.46;
535.00 503.57;
540.00 513.90;
545.00 444.04;
550.00 351.74;
555.00 293.87;
560.00 252.11;
565.00 220.88;
570.00 198.03;
575.00 181.37;
580.00 168.86;
585.00 159.09;
590.00 151.18;
595.00 144.59;
600.00 138.96];

# ╔═╡ f46ff2e5-f715-4fac-8a4d-3d80fe6d8a3d
md"""
Although this is a bit of an extreme example, as we can see here, by over-estimating the critical temperature by a few kelvin, PC-SAFT and CPA dramatically overestimate the value of the heat capacity. This is because the heat capacity diverges as we approach the critical temperature; this is why it is so important to predict this point correctly. Furthermore, in the above, we can see the second improvement made by SAFT-VR Mie: in contrast to the other equations of state, it replicates the experimental data to a higher degree of accuracy, particularly in the liquid phase. One can attribute this to SAFT-VR Mie being a more-physically sound theory. 

The last thing to consider is how the SAFT equations handle mixtures:
"""

# ╔═╡ 7adfcef4-52e1-4bf9-b110-1df48cce30dc
Exp_MeB = [0.0483 0.2719 139.5;
0.0943 0.3890 163.0;
0.1589 0.4873 188.7;
0.2967 0.5965 223.4;
0.4449 0.6645 244.5;
0.5721 0.7129 255.5;
0.6775 0.7487 261.7;
0.8461 0.8328 267.5;
0.9176 0.8920 265.7;
0.9509 0.9287 260.5];

# ╔═╡ c71b68c7-c0a4-409d-adec-542b01e88ce9
md"""
As we can see above, SAFT equations can reproduce experimental data far better than the cubics, even with fitted parameters. What is also noticeable is that all three SAFT equations have similar predictions. This is expected as we are, again, fitting the binary-interaction parameter, $k_{ij}$, using experimental data. However, because of the more-physical model provided by the SAFT equations, we are able to more-accurately capture the experimental data. Nevertheless, the above example serves to show that, for the purposes of vapour--liquid equilibria, it is quite difficult to distinguish between the different equations.

All that being said, it may seem from the above that the only discernable difference between these SAFT equations is the ability to model the critical point and bulk properties, where SAFT-VR Mie and SAFT-$\gamma$ Mie have the advantage. However, one needs to consider a few things. Firstly, it is best practice to avoid the critical point for most species. Although the topic of critical phenomena is beyond the scope of this course, we note that there are large fluctuations in most properties near the critical point. As such, in any case, we should avoid operating near the critical point. Furthermore, in terms of relative errors, at sub-critical conditions, PC-SAFT and CPA are not significantly worse than SAFT-VR Mie and SAFT-$\gamma$ Mie. 

Finally, in simulating processes, our codes need to be very fast and, unfortunately, due to their more-complex nature, SAFT-VR Mie and SAFT-$\gamma$ Mie are at the disadvantage here:
"""

# ╔═╡ ee7ea366-e3bd-41ee-900a-1244988f64c5
md"""
* PC-SAFT:
"""

# ╔═╡ 1d9dba36-3383-41c4-aa03-2b8edac67d10
md"""
* CPA:
"""

# ╔═╡ 04e034e5-fb91-40a8-b442-0e1c6d7320b7
md"""
* SAFT-VR Mie:
"""

# ╔═╡ f7d75eba-0c05-458e-b720-eee8a401df9d
md"""
Depending on the device you are using and species you are modelling, SAFT-VR Mie can be up to three times slower than PC-SAFT and five times slower than CPA! As such, we do need to weigh the advantages of these equations of state carefully, both in terms of accuracy and computational resources.
"""

# ╔═╡ 0311216d-53ef-4f61-81fa-9f9a2a3a3456
md"""
### Footnotes
[^1]: These two $m_i$ are different but the developers of CPA choose to re-use $m_i$ for the $\alpha$-function as this is also the standard for cubics.
"""

# ╔═╡ c6b6ff0e-867c-48f4-8547-eaf5a3f27740
Exp_water = [175.61 1.8636e-07 28230. 3.5423e-05;
185.61 8.7675e-07 27922. 3.5814e-05;
195.61 3.4849e-06 27610. 3.6219e-05;
205.61 1.2000e-05 27298. 3.6632e-05;
215.61 3.6536e-05 26990. 3.7051e-05;
225.61 0.00010003 26684. 3.7475e-05;
235.61 0.00024970 26382. 3.7905e-05;
245.61 0.00057493 26082. 3.8341e-05;
255.61 0.0012330 25784. 3.8783e-05;
265.61 0.0024832 25489. 3.9232e-05;
275.61 0.0047301 25196. 3.9689e-05;
285.61 0.0085732 24904. 4.0154e-05;
295.61 0.014863 24612. 4.0630e-05;
305.61 0.024759 24319. 4.1120e-05;
315.61 0.039790 24023. 4.1626e-05;
325.61 0.061904 23724. 4.2152e-05;
335.61 0.093523 23418. 4.2702e-05;
345.61 0.13758 23105. 4.3280e-05;
355.61 0.19755 22783. 4.3893e-05;
365.61 0.27748 22449. 4.4546e-05;
375.61 0.38200 22101. 4.5247e-05;
385.61 0.51632 21737. 4.6005e-05;
395.61 0.68626 21353. 4.6831e-05;
405.61 0.89825 20947. 4.7739e-05;
415.61 1.1593 20514. 4.8746e-05;
425.61 1.4773 20050. 4.9876e-05;
435.61 1.8603 19547. 5.1158e-05;
445.61 2.3172 18999. 5.2636e-05;
455.61 2.8566 18393. 5.4368e-05;
465.61 3.4867 17715. 5.6449e-05;
475.61 4.2163 16938. 5.9040e-05;
485.61 5.0599 16008. 6.2468e-05;
495.61 6.0437 14793. 6.7600e-05;
505.61 7.1915 12982. 7.7029e-05;
175.61 1.8635e-07 0.00012764 7834.4;
185.61 8.7679e-07 0.00056829 1759.7;
195.61 3.4849e-06 0.0021438 466.45;
205.61 1.2000e-05 0.0070261 142.33;
215.61 3.6536e-05 0.020415 48.984;
225.61 0.00010003 0.053470 18.702;
235.61 0.00024970 0.12799 7.8132;
245.61 0.00057493 0.28321 3.5309;
255.61 0.0012330 0.58495 1.7096;
265.61 0.0024832 1.1370 0.87954;
275.61 0.0047301 2.0942 0.47750;
285.61 0.0085732 3.6777 0.27191;
295.61 0.014863 6.1898 0.16156;
305.61 0.024759 10.030 0.099702;
315.61 0.039790 15.711 0.063651;
325.61 0.061904 23.874 0.041887;
335.61 0.093523 35.308 0.028323;
345.61 0.13758 50.967 0.019621;
355.61 0.19755 71.996 0.013890;
365.61 0.27748 99.767 0.010023;
375.61 0.38200 135.92 0.0073573;
385.61 0.51632 182.44 0.0054812;
395.61 0.68626 241.76 0.0041364;
405.61 0.89825 316.90 0.0031556;
415.61 1.1593 411.68 0.0024290;
425.61 1.4773 530.97 0.0018833;
435.61 1.8603 680.67 0.0014691;
445.61 2.3172 867.19 0.0011532;
455.61 2.8566 1096.2 0.00091226;
465.61 3.4867 1374.1 0.00072777;
475.61 4.2163 1720.9 0.00058110;
485.61 5.0599 2211.0 0.00045228;
495.61 6.0437 2958.8 0.00033797;
505.61 7.1915 4240.2 0.00023584];

# ╔═╡ d4d5b02c-fd60-4bcb-bec4-6d8380e00b67
begin
	N = 150
	function Clapeyron.x0_crit_pure(model::Clapeyron.CPAModel)
	    lb_v = Clapeyron.lb_volume(model)
	    if isempty(model.params.epsilon_assoc.values[1,1])
	        [2.0, log10(lb_v/0.3)]
	    else
	        [2.5, log10(lb_v/0.3)]
	    end
	end
	mw1 = PR(["methanol"];idealmodel=WalkerIdeal)
	mw2 = PCSAFT(["methanol"];idealmodel=WalkerIdeal)
	mw3 = CPA(["methanol"];idealmodel=WalkerIdeal)
	mw4 = SAFTVRMie(["methanol"];idealmodel=WalkerIdeal)

	Tcw1,pcw1,vcw1 = crit_pure(mw1)
	Tcw2,pcw2,vcw2 = crit_pure(mw2)
	Tcw3,pcw3,vcw3 = crit_pure(mw3)
	Tcw4,pcw4,vcw4 = crit_pure(mw4)

	Tw1 = range(150.,Tcw1,length=N)
	Tw2 = range(150.,Tcw2,length=N)
	Tw3 = range(150.,Tcw3,length=N)
	Tw4 = range(150.,Tcw4,length=N)

	satw1 = saturation_pressure.(mw1,Tw1)
	satw2 = saturation_pressure.(mw2,Tw2)
	satw3 = saturation_pressure.(mw3,Tw3)
	satw4 = saturation_pressure.(mw4,Tw4)

	pw1 = [satw1[i][1] for i ∈ 1:N]
	pw2 = [satw2[i][1] for i ∈ 1:N]
	pw3 = [satw3[i][1] for i ∈ 1:N]
	pw4 = [satw4[i][1] for i ∈ 1:N-2]
	append!(pw4,pcw4)

	vlw1 = [satw1[i][2] for i ∈ 1:N]
	vlw2 = [satw2[i][2] for i ∈ 1:N]
	vlw3 = [satw3[i][2] for i ∈ 1:N]
	vlw4 = [satw4[i][2] for i ∈ 1:N-2]

	vvw1 = [satw1[i][3] for i ∈ 1:N]
	vvw2 = [satw2[i][3] for i ∈ 1:N]
	vvw3 = [satw3[i][3] for i ∈ 1:N]
	vvw4 = [satw4[i][3] for i ∈ 1:N-2]
	append!(vlw4,reverse(vvw4))

	Tw5 = [Tw4[i] for i ∈ 1:N-2]
	append!(Tw5,reverse(Tw5))

	plot(1e-3./vlw1,Tw1,color=:blue,ylim=(150,550),
		title="Vapour-liquid envelope of methanol",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vvw1,Tw1,color=:blue,label="")
	plot!(1e-3./vlw2,Tw2,color=:red,
		label="PC-SAFT")
	plot!(1e-3./vvw2,Tw2,color=:red,label="")
	plot!(1e-3./vlw3,Tw3,color=:green,
		label="CPA")
	plot!(1e-3./vvw3,Tw3,color=:green,label="")
	plot!(1e-3./vlw4,Tw5,color=:purple,
		label="SAFT-VR Mie")
	# plot!(1e-3./vvw4,Tw5,color=:purple,label="")
	scatter!(Exp_water[:,3]*1e-3,Exp_water[:,1],color=:white,edgecolor=:blue,label="")
	scatter!([8.6],[512.6],color=:black,edgecolor=:blue,label="")

end

# ╔═╡ e3dcda2a-4ef9-46b0-a63c-8383eb247f3d
begin
	p = 1.2e7

	T = range(450,600,length=N)

	Cp1 = isobaric_heat_capacity.(mw1,p,T)
	Cp2 = isobaric_heat_capacity.(mw2,p,T)
	Cp3 = isobaric_heat_capacity.(mw3,p,T)
	Cp4 = isobaric_heat_capacity.(mw4,p,T)

	plot(T,Cp1,xlim=(450,600),ylim=(0,600),
		title="Isobaric heat capacity of methanol with different EoS",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,
		ylabel=L"C_p / (\mathrm{J/K/mol})", xlabel=L"T / K",color=:blue)
	plot!(T,Cp2,label="PC-SAFT",color=:red)
	plot!(T,Cp3,label="CPA",color=:green)
	plot!(T,Cp4,label="SAFT-VR Mie",color=:purple)
	scatter!(Exp_MeOH[:,1],Exp_MeOH[:,2],color=:white,edgecolor=:blue,label="")
	annotate!(455, 550., text("p=12. MPa", :black, :left, 14))

end

# ╔═╡ d1bcf93b-03ed-4aa9-b5c1-61d30691a4dc
begin
	mix_bm1 = PR(["benzene","methanol"];alpha=TwuAlpha,userlocations=["./assets"])
	mix_bm2 = PCSAFT(["benzene","methanol"])
	mix_bm3 = CPA(["benzene","methanol"])
	mix_bm4 = SAFTgammaMie([("benzene",["aCH"=>6]),"methanol"];userlocations=["./assets"])

	x = range(0.,1.,length=N)
	X2 = Clapeyron.Fractions.FractionVector.(x)

	T_mix_bm = 433.15
	
	bub_bm1  = bubble_pressure.(mix_bm1,T_mix_bm,X2)
	bub_bm2  = bubble_pressure.(mix_bm2,T_mix_bm,X2)
	bub_bm3  = bubble_pressure.(mix_bm3,T_mix_bm,X2)
	bub_bm4 = bubble_pressure.(mix_bm4,T_mix_bm,X2)

	p_bm1 = [bub_bm1[i][1] for i ∈ 1:N]
	y_bm1 = [bub_bm1[i][4][1] for i ∈ 1:N]
	p_bm2 = [bub_bm2[i][1] for i ∈ 1:N]
	y_bm2 = [bub_bm2[i][4][1] for i ∈ 1:N]
	p_bm3 = [bub_bm3[i][1] for i ∈ 1:N]
	y_bm3 = [bub_bm3[i][4][1] for i ∈ 1:N]
	p_bm4 = [bub_bm4[i][1] for i ∈ 1:N]
	y_bm4 = [bub_bm4[i][4][1] for i ∈ 1:N]

	plot(x,p_bm1./1e6,color=:blue,xlim=(0,1),
		title="pxy diagram of benzene+methanol",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:bottomleft,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y_bm1,p_bm1./1e6,color=:blue,
		label="")
	plot!(x,p_bm2./1e6,color=:red,
		label="PC-SAFT")
	plot!(y_bm2,p_bm2./1e6,color=:red,
		label="")
	plot!(x,p_bm3./1e6,color=:green,
		label="CPA")
	plot!(y_bm3,p_bm3./1e6,color=:green,
		label="")
	plot!(x,p_bm4./1e6,color=:purple,
		label="SAFT-γ Mie")
	plot!(y_bm4,p_bm4./1e6,color=:purple,
		label="")
	scatter!(1 .-Exp_MeB[:,1],Exp_MeB[:,3].*0.00689476,label="Experimental",color=:white,edgecolor=:blue)
	scatter!(1 .-Exp_MeB[:,2],Exp_MeB[:,3].*0.00689476,label="",color=:white,edgecolor=:blue)
	annotate!(0.75, 1.82, text("T=433.15 K", :black, :left, 14))

end

# ╔═╡ b330c503-4a05-4cab-9c92-1fc3e621096c
@benchmark isobaric_heat_capacity(mw2,1e5,298.15)

# ╔═╡ f986c852-f3e4-4c22-9dc5-7d536f180b20
@benchmark isobaric_heat_capacity(mw3,1e5,298.15)

# ╔═╡ ddd504a8-b907-4a33-a69a-be30316cf3ea
@benchmark isobaric_heat_capacity(mw4,1e5,298.15)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
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

[compat]
BenchmarkTools = "~1.3.1"
Clapeyron = "~0.3.7"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Optim = "~1.7.0"
Plots = "~1.31.4"
PlutoUI = "~0.7.39"
PolynomialRoots = "~1.0.0"
Roots = "~2.0.2"
ShortCodes = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "9fe256970afcfaff39be4e01821711b5a222a3b4"

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
version = "1.1.1"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "8d9e48436c5589fbd51ae8c8165a299a219188c0"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.15"

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
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

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
git-tree-sha1 = "7c88f63f9f0eb5929f15695af9a4d7d3ed278a91"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.16"

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
git-tree-sha1 = "9f4f5a42de3300439cb8300236925670f844a555"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.1"

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
version = "1.2.0"

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

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

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
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "77172cadd2fdfa0c84c87e3a01215a4ca7723310"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.0.0"

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
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─061fb273-ec2a-4e89-ba73-c4d3c4a76176
# ╟─f9c5a11e-0e45-11ed-2b32-153050d16a4a
# ╟─9496586f-63f2-4cb8-9ece-7d571dd68d4e
# ╟─5e449695-f7b2-452c-82b2-af193e776280
# ╟─ae398071-8969-4b6d-ba6c-12122c2c5b94
# ╟─762c0786-9152-429f-906f-728056200a9a
# ╟─2609eb20-7f6c-455b-a1b3-f4cb99d1e50c
# ╟─25437516-b88b-478c-b6ab-be0a293305de
# ╠═0b3545ea-ab28-4de3-9be2-91d3a6cccea8
# ╟─fe094bfb-65a7-4549-9db9-d27ece7c79a5
# ╟─704ed6c5-6730-42a9-bdbc-6b04e4b666b7
# ╟─02f3060e-df47-46a3-b1fa-37f10959b86a
# ╠═49a74e5c-c17b-40a8-8bc4-f62ae3131fc3
# ╟─c76a488b-d93c-403c-8ebf-44ca79bbd849
# ╟─1aed05c7-68e9-466c-a449-3ad6c3fdebb9
# ╟─d2dedb47-6d78-40a7-8b47-85d86ae2fa7a
# ╟─9a2ed4e6-a39a-4e92-b075-cfc6b62b3795
# ╟─da387b61-a0f2-496f-be79-eccf3d9394cc
# ╟─c1a79228-e556-4d6e-9f1d-d88efc1747ed
# ╟─303fba91-9844-4649-82be-e169d6f52a9c
# ╟─89ff71b7-130d-4642-9558-e49efa51247e
# ╟─19f044fc-c181-44d8-bf4f-3609eade334c
# ╟─76959b4f-ae28-4b9a-b24e-572fccced269
# ╟─909b14ac-5b9d-40d9-959b-9b6c171d1b3a
# ╟─cb1b6005-39c0-4247-b824-65a25b79dc33
# ╟─76790f3f-b101-48d8-b6b0-fc1df6e14e0f
# ╟─b13b2c8c-6366-43e9-936b-6deb67be6db1
# ╟─44256338-ce74-448a-a5e2-5bca38e3c2e0
# ╟─c8de4d65-190b-4f36-b9ea-82bfebf9c632
# ╟─c0eb6b68-29b1-413f-a38e-c6781f9c4552
# ╟─716676d5-b6f5-4932-b5c7-b166a2f8faa3
# ╟─a1dad25b-453c-4689-b04c-39a51d1c90b8
# ╟─0ba8192f-5e63-4fb7-990e-272031e99d21
# ╟─dc05c788-96f9-4c4f-b7bb-5e007e6fd9ee
# ╟─9b3e9913-7109-4cdc-901f-74191a3aa1b2
# ╟─d0d9ed02-b51c-4d6f-8bd2-4c89041a3b50
# ╟─66ebf47c-aecc-42d6-b67d-8837d5086a44
# ╟─d4d5b02c-fd60-4bcb-bec4-6d8380e00b67
# ╟─dda0f661-d548-4022-9002-b1507bb998a1
# ╟─e3dcda2a-4ef9-46b0-a63c-8383eb247f3d
# ╟─7742a363-e2a0-4792-bc07-26f3650bb9d0
# ╟─f46ff2e5-f715-4fac-8a4d-3d80fe6d8a3d
# ╟─d1bcf93b-03ed-4aa9-b5c1-61d30691a4dc
# ╟─7adfcef4-52e1-4bf9-b110-1df48cce30dc
# ╟─c71b68c7-c0a4-409d-adec-542b01e88ce9
# ╟─ee7ea366-e3bd-41ee-900a-1244988f64c5
# ╟─b330c503-4a05-4cab-9c92-1fc3e621096c
# ╟─1d9dba36-3383-41c4-aa03-2b8edac67d10
# ╟─f986c852-f3e4-4c22-9dc5-7d536f180b20
# ╟─04e034e5-fb91-40a8-b442-0e1c6d7320b7
# ╟─ddd504a8-b907-4a33-a69a-be30316cf3ea
# ╟─f7d75eba-0c05-458e-b720-eee8a401df9d
# ╟─0311216d-53ef-4f61-81fa-9f9a2a3a3456
# ╟─c6b6ff0e-867c-48f4-8547-eaf5a3f27740
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
