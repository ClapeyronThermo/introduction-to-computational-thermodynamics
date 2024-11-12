### A Pluto.jl notebook ###
# v0.20.0

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
# Section 2.4 – Statistical Association Fluid Theory
_N.B.: This will be a high-level overview of the SAFT equations, with more emphasis on the physical picture than the implementation. If you were able to implement the generalised cubic equation in the previous section, you should be able to implement any of the SAFT equations (with the exception of one aspect which will be discussed here)._

We previously established that, while the cubics are some of the most flexible equations of state, the range of systems and properties you can model accurately with them is limited. Taking a few steps back, we mentioned previously that the van der Waals equation can actually be derived analytically. Writing out the residual Helmholtz free energy

$$A_\mathrm{res.} = -nRT\log{(1-nb/V)}-\frac{n^2a}{V},$$

we can imagine that the first term represents the contribution from the presence of particles as hard spheres ($A_\mathrm{HS}$) and the second is a perturbation from those hard spheres to account for dispersive, pair-wise interactions ($A_1$). Thus, we could write it out as

$$A_\mathrm{res.} = A_\mathrm{HS}+A_1.$$

Note that $A_\mathrm{HS}$ has been refined in literature to better model true hard-sphere systems. Nevertheless, one way to improve upon this equation is to take higher-order perturbations from the hard-sphere model to account for many-body interactions

$$A_\mathrm{res.} = A_\mathrm{HS}+A_1+\frac{A_2}{Nk_\mathrm{B}T}+….$$

In such approaches, species are no longer characterised by the parameters $a$ and $b$, we now use the diameter of our species ($\sigma$) and the dispersive energy parameter ($\epsilon$). Visually:
"""

# ╔═╡ 9496586f-63f2-4cb8-9ece-7d571dd68d4e
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Seg.svg" height="300"></center>""")

# ╔═╡ 5e449695-f7b2-452c-82b2-af193e776280
md"""
Despite these improvements, it doesn't change the fact that we are just modelling spherical systems with dispersion interactions, albeit more accurately; most species will not fit this description. For decades after van der Waals derived his equation, researchers have developed new approaches to more accurately model a larger range of species. In 1989, Chapman _et al._ published the Statistical Associating Fluid Theory (SAFT) where they first grouped the hard-sphere and perturbation contributions into a single term (the segment term, $A_\mathrm{seg.}$) and introduced two new terms

$$A_\mathrm{res.} = A_\mathrm{seg.}+A_\mathrm{chain}+A_\mathrm{assoc.}.$$

### Chain term
The first new term is the chain term. Here, we account for the formation of a chain made up of $m$ segments (thus introducing one new parameter). This term allows us to better model chain-like species (anything from alkanes to polymers).
"""

# ╔═╡ ae398071-8969-4b6d-ba6c-12122c2c5b94
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Chain.svg" height="400"></center>""")

# ╔═╡ 762c0786-9152-429f-906f-728056200a9a
md"""
One thing to bear in mind is that this new $m$ parameter, the number of segments doesn't necessarily represent the number of monomers (e.g., number of CH₂ groups in alkanes) and doesn't even have to be an integer (in this case, a non-integer value can be imagined as a "merging" of spheres). This $m$ is just a "best-fit" for the number of hard spheres to represent a molecule.

### Association term
The second term introduced by Chapman _et al._ is the association term. Here, species are modelled as having small, tangential sites which can overlap with the sites from other species to form a dimer.
"""

# ╔═╡ 2609eb20-7f6c-455b-a1b3-f4cb99d1e50c
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Assoc.svg" height="400"></center>""")

# ╔═╡ 25437516-b88b-478c-b6ab-be0a293305de
md"""
The idea here is that this approach can be used to mimic hydrogen bonding in systems like water. This interaction is characterised by an association energy parameter ($\epsilon^\mathrm{assoc.}_{ij,ab}$) and a bonding "volume" (either dimensionless, $\kappa_{ij,ab}$ or non-dimensionless, $K_{ij,ab}$).

Whilst the segment and chain terms are tedious, but explicit equations which simply take a lot of time to code, the association term introduces one additional level of complexity. At the core of the association term is the association fraction of a site $a$ on species $i$, $X_{i,a}$, which is given by

$$X_{ia} = \frac{1}{1 +\sum_{j,b}{\rho_{j}\Delta_{ijab}X_{jb}}},$$

where $\Delta_{ij,ab}$ is the association strength between site $a$ on species $i$ and site $b$ on species $j$. This equation, at first, appears a bit daunting. If we simplify it so that we only have one species with one site which can only bond with itself, we have

$$X = \frac{1}{1 +\rho \Delta X}.$$

The issue should now become more apparent: this equation is implicit (i.e. $y=f(x,y)$). This means we will need to use an iterative method to solve for the association fraction $X$. Thankfully, in the above case, the solution can actually be solved for explicitly as

$$X = \frac{-1+\sqrt{1+4\rho\Delta}}{2\rho\Delta}.$$

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
The issue with this approach is that it oscillates around the true solution.
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
One easy solution is to introduce a damping factor where, rather than accepting the new solution, we "damp it" using some of our current solution

$$X^{(i+1)}=\alpha X^{(i)}+(1-\alpha)f(X^{(i)}).$$

After much experimentation in literature, the optimal damping factor, $\alpha$, is found to be 0.5. This is enough to eliminate the oscillations but still converge relatively quickly.
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
The SAFT equations are derived based on pure-component systems. As such, like the cubics, SAFT equations need to use mixing rules. Whereas with cubics, we applied mixing rules to the parameters (which we do with some SAFT equations), for most SAFT equations we apply mixing rules on the Helmholtz free energy directly. This is handled on a term-by-term basis. Firstly, in the segment term, a quadratic mixing rule is used

$$\bar{A}_\mathrm{seg.}=\sum_i\sum_jx_ix_j A_{\mathrm{seg.},ij},$$

where, for $i=j$, $A_{\mathrm{seg.},ii}$ is just the segment contribution for a pure component $i$. For $i\neq j$, $A_{\mathrm{seg.},ij}$ now represents the contribution from the fictitious fluid which is characterised by parameters

$$\sigma_{ij}=\frac{\sigma_{ii}+\sigma_{jj}}{2}(1-l_{ij})$$
$$\epsilon_{ij}=\sqrt{\epsilon_{ii}\epsilon_{jj}}(1-k_{ij}).$$

Again, we have our "fudge factors" $l_{ij}$ and $k_{ij}$, which, like the cubics, must be fit using experimental data.

For the chain term, a linear mixing rule is used instead, which requires no new parameters

$$\bar{A}_\mathrm{chain}=\sum_ix_i A_{\mathrm{chain},i}.$$

The association term is far more complex as we can have association between sites on different species (e.g. hydrogen bonding between water and methanol). These are all accounted for in the full equation for the association fraction above.
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
* The "bonding" volume: $\kappa^\mathrm{assoc.}_{ij,ab}$ or $K^\mathrm{assoc.}_{ij,ab}$
Note that the last two parameters are only needed for species which experience association. Unfortunately, while the parameters for the cubics can easily be obtained using information from the critical point and acentricity, this is not possible for SAFT equations. The pure parameters are usually obtained by regressing them using experimental data (usually the saturation pressure and saturated liquid densities). Thus, despite its strong physical foundation, SAFT equations are still limited by available experimental data (there are exceptions to this which will be shown later). Thankfully, large databases of parameters are available online which should allow one to model most species of interest.

Before going online to search for parameters, there is one key detail to keep in mind: there are many different variants of SAFT equations. Whilst the formalism described above is generally true, some minor differences will be described below. We will focus on the more popular and commonly used SAFT equations.
"""

# ╔═╡ 303fba91-9844-4649-82be-e169d6f52a9c
md"""
## Section 2.4.1 – Perturbed-Chain SAFT (PC-SAFT)
"""

# ╔═╡ 89ff71b7-130d-4642-9558-e49efa51247e
md"""
Possibly the most popular variant of the SAFT equation, Perturbed-Chain SAFT (PC-SAFT) was developed by Gross and Sadowski. The key difference in this equation is primarily how the terms are arranged

$$A_\mathrm{res.} = A_\mathrm{HC}+A_\mathrm{disp.}+A_\mathrm{assoc.}.$$

Here, we start with a hard-chain (HC) reference system. All this, really, is the hard-sphere term and standard chain term summed in a single term ($A_\mathrm{HC}=A_\mathrm{HS}+A_\mathrm{chain}$). The big change comes from the fact that, rather than perturbing the hard-sphere system and then forming the chain, in PC-SAFT, we do it the other way around: the chain is formed first and then this system is perturbed to account for dispersive interactions ($A_\mathrm{disp.}$). This difference is quite subtle, but, it does result in a more physically sound model. The association term is unchanged from the original formulation. Visually:
"""

# ╔═╡ 19f044fc-c181-44d8-bf4f-3609eade334c
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/PCSAFT.svg" height="500"></center>""")

# ╔═╡ 76959b4f-ae28-4b9a-b24e-572fccced269
md"""
The primary reasons behind PC-SAFT's popularity are threefold. For one, the code for PC-SAFT was available open-source from publication. Secondly, there is an abundance of parameters available (over 250 species), including binary interaction parameters. Finally, many variants of the PC-SAFT equation have been developed. This last point, unfortunately, is actually one of the downsides of PC-SAFT: one has to be very careful which version of PC-SAFT is being used. Just to name a few:
* Polar PC-SAFT (PPC-SAFT): An additional term is added to the PC-SAFT equation to account for dipole interactions. With it, we introduce a new parameter, $\mu_i$, the dipole moment of a segment. The overall equation is
$$A_\mathrm{res.} = A_\mathrm{HC}+A_\mathrm{disp.}+A_\mathrm{assoc.}+A_\mathrm{polar}.$$
* Group-contribution PC-SAFT (GC-PC-SAFT): A group-contribution method for PC-SAFT which allows users to model a larger range of species. Unfortunately, there are different ways group-contribution methods are handled in PC-SAFT, so one still needs to be consistent.
* Critical point PC-SAFT (CP-PC-SAFT): As a way to become more like a cubic, with this PC-SAFT variant, the parameters are based on the critical point.
* Simplified PC-SAFT (sPC-SAFT): An attempt to reduce computational costs, with this version, many of the equations used in the standard PC-SAFT equation are simplified.
Furthermore, as a warning when one looks for parameters, implementations exist in which the above approaches are mixed. Generally speaking, PC-SAFT alone is most commonly used.
"""

# ╔═╡ 909b14ac-5b9d-40d9-959b-9b6c171d1b3a
md"""
## Section 2.4.2 – Cubic plus association (CPA)
"""

# ╔═╡ cb1b6005-39c0-4247-b824-65a25b79dc33
md"""
As the name implies, the idea behind the cubic plus association (CPA) equation of state is that most species can be modelled quite accurately using a cubic; only associating species are problematic. Thus, what if we keep the residual term from the cubic but add the association term at the end? This gives us a free energy of the form

$$A_\mathrm{CPA} = A_\mathrm{SRK}+A_\mathrm{assoc.}.$$

From an alternative point of view, we are replacing the chain and segment terms with a cubic equation of state. This simplifies things in terms of the parameters as, for non-associating species, we can still use the parameters we would use in our original cubic. For associating species, we replace the $\sigma_{ii}$, $\epsilon_{ii}$ and $m_i$ parameters with $a_{ii}$, $b_{ii}$ and $m_i$[^1], where $m_i$ is used in a new $\alpha$ function

$$\alpha_i = \left( 1+m_i \left( 1-\sqrt{T/T_c} \right) \right)^2.$$

This equation has two key benefits:
1. We only need to fit parameters for associating species (water, methanol, etc.).
2. All the tools we developed for cubics (new $\alpha$ functions, volume translation methods, advanced mixing rules, etc.) can be applied to CPA.

This makes CPA a very popular alternative to other SAFT equations because of its close link to cubics. Naturally, because it loses the physical foundation that SAFT equations are based on, it can't be used to model as wide a range of species and properties as accurately as SAFT equations can.
"""

# ╔═╡ 76790f3f-b101-48d8-b6b0-fc1df6e14e0f
md"""
## Section 2.4.3 – SAFT-VR Mie
"""

# ╔═╡ b13b2c8c-6366-43e9-936b-6deb67be6db1
md"""
The Variable-Range SAFT (SAFT-VR) framework was originally developed by Gil-Villegas _et al._. Previously, with most SAFT equations, it is assumed that a Lennard–Jones potential could be used to represent the dispersive interactions. In SAFT-VR, the idea was to use potentials where one could vary the shape of the potential. The simplest of these was the square-well potential where a new potential shape parameter ($\lambda$) was introduced, which measured the range of the dispersive interactions.
"""

# ╔═╡ 44256338-ce74-448a-a5e2-5bca38e3c2e0
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/SW.svg" height="500"></center>""")

# ╔═╡ c8de4d65-190b-4f36-b9ea-82bfebf9c632
md"""
However, this did not really improve the modelling of species much. It wasn't until Lafitte _et al._ extended this concept of a variable-range potential to a Mie potential that significant improvements were realised. This introduced two new potential shape parameters, $\lambda_\mathrm{a}$ (characterising the attractive part) and $\lambda_\mathrm{r}$ (characterising the repulsive part).
"""

# ╔═╡ c0eb6b68-29b1-413f-a38e-c6781f9c4552
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Mie.svg" height="500"></center>""")

# ╔═╡ 716676d5-b6f5-4932-b5c7-b166a2f8faa3
md"""
This led to the development of the SAFT-VR Mie equation of state. At the surface, the formulation is generally the same as the standard SAFT equation

$$A_\mathrm{res.} = A_\mathrm{seg.}+A_\mathrm{chain}+A_\mathrm{assoc.}.$$

However, internally, two big changes were made:
* The segment term was taken to a higher-order perturbation. Up to this point, all SAFT equations only went up to a second-order perturbation expansion. For SAFT-VR Mie, it was decided to go up to third order. Physically, this means that many-body interactions in SAFT-VR Mie are better characterised. Practically, it improved the modelling of properties near the critical point and the modelling of bulk properties in general.
* Similar to PC-SAFT, while we do perturb the hard-sphere system first and then form a chain, for SAFT-VR Mie, in effect, a perturbation of the chain is also applied. This is a more theoretically consistent implementation of the chain term.
These two modifications have led SAFT-VR Mie to become, possibly, the most advanced equation of state to date. Unfortunately, because of the introduction of two additional parameters ($\lambda_\mathrm{a}$ and $\lambda_\mathrm{r}$), it does mean more parameters need to be fitted (although, generally, it is assumed that $\lambda_\mathrm{a}=6$). As it is slightly newer than PC-SAFT, not as many parameters have been fitted for SAFT-VR Mie, limiting its range of applicability. However, a group-contribution method for SAFT-VR Mie was also developped, which will be discussed next.
"""

# ╔═╡ a1dad25b-453c-4689-b04c-39a51d1c90b8
md"""
## Section 2.4.4 – Heterosegmented SAFT-VR Mie (SAFT-$\gamma$ Mie)
"""

# ╔═╡ 0ba8192f-5e63-4fb7-990e-272031e99d21
md"""
To be more widely applicable, rather than fit many parameters in SAFT-VR Mie, the developers decided to instead focus on developing a group-contribution method, known as SAFT-$\gamma$ Mie. However, another modification was made to SAFT-$\gamma$ Mie that further improved its ability to model a wider range of species. In both PC-SAFT and SAFT-VR Mie, it is assumed that the segments in a chain for a given molecule all have the same size. As the chains in SAFT-$\gamma$ Mie are assembled of groups, which can have different sizes, this limitation does not apply to SAFT-$\gamma$ Mie. This is why it can often be thought of as a heterosegmented version of SAFT-VR Mie. Visually:
"""

# ╔═╡ dc05c788-96f9-4c4f-b7bb-5e007e6fd9ee
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/SAFTMie.svg" height="500"></center>""")

# ╔═╡ 9b3e9913-7109-4cdc-901f-74191a3aa1b2
md"""
Unfortunately, despite this improved physical picture, the approach does still suffer from the limitations of a group-contribution approach. Nevertheless, with over 50 groups so far, SAFT-$\gamma$ Mie is often thought of as the state-of-the-art SAFT equation of state.
"""

# ╔═╡ d0d9ed02-b51c-4d6f-8bd2-4c89041a3b50
md"""
## Section 2.4.5 – Comparing SAFT equations
"""

# ╔═╡ 66ebf47c-aecc-42d6-b67d-8837d5086a44
md"""
Having now covered most of the SAFT equations one might encounter, one question remains: which one should you use? Unsurprisingly, the answer is: it depends. Speaking in generalities, most SAFT models, if parameterised correctly, will perform almost equivalently. For example, consider methanol below.
"""

# ╔═╡ dda0f661-d548-4022-9002-b1507bb998a1
md"""
There is certainly a substantial improvement in the predictions of liquid densities using any of the SAFT models, especially when compared to a standard cubic like Peng–Robinson (PR). However, a more subtle detail appears when we approach the critical point: while the cubic capture exactly the critical temperature, all SAFT equations overestimate it, due to how the parameters are fitted. This is something inherent to all perturbation-based theories. With SAFT-VR Mie, we attempt to address this by extending its perturbation to third order. Even then, the critical temperature is overestimated by a few Kelvin. Surprisingly, even a small difference in the estimation of the critical temperature can have a dramatic impact on bulk properties of our species.
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
Although this is a bit of an extreme example, as we can see here, by overestimating the critical temperature by a few kelvin, PC-SAFT and CPA dramatically overestimate the value of the heat capacity. This is because the heat capacity diverges as we approach the critical temperature; this is why it is so important to predict this point correctly. Furthermore, in the above, we can see the second improvement made by SAFT-VR Mie: in contrast to the other equations of state, it replicates the experimental data to a higher degree of accuracy, particularly in the liquid phase. One can attribute this to SAFT-VR Mie being a more physically sound theory.

The last thing to consider is how the SAFT equations handle mixtures.
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
As we can see above, SAFT equations can reproduce experimental data far better than the cubics, even with fitted parameters. What is also noticeable is that all three SAFT equations have similar predictions. This is expected as we are, again, fitting the binary-interaction parameter, $k_{ij}$, using experimental data. However, because of the more physical model provided by the SAFT equations, we are able to more accurately capture the experimental data. Nevertheless, the above example serves to show that, for the purposes of vapour–liquid equilibria, it is quite difficult to distinguish between the different equations.

All that being said, it may seem from the above that the only discernable difference between these SAFT equations is the ability to model the critical point and bulk properties, where SAFT-VR Mie and SAFT-$\gamma$ Mie have the advantage. However, one needs to consider a few things. Firstly, it is best practice to avoid the critical point for most species. Although the topic of critical phenomena is beyond the scope of this course, we note that there are large fluctuations in most properties near the critical point. As such, in any case, we should avoid operating near the critical point. Furthermore, in terms of relative errors, at sub-critical conditions, PC-SAFT and CPA are not significantly worse than SAFT-VR Mie and SAFT-$\gamma$ Mie.

Finally, in simulating processes, our codes need to be very fast and, unfortunately, due to their more complex nature, SAFT-VR Mie and SAFT-$\gamma$ Mie are at the disadvantage here.
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
[^1]: These two $m_i$ are different but the developers of CPA choose to re-use $m_i$ for the $\alpha$ function as this is also the standard for cubics.
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
		ylabel=L"C_p / (\mathrm{J/(mol \cdot K)})", xlabel=L"T / \mathrm{K}",color=:blue)
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
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "9fe256970afcfaff39be4e01821711b5a222a3b4"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "d80af0733c99ea80575f612813fa6aa71022d33a"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.0"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3640d077b6dafd64ceb8fd5c1ec76f7ca53bcf76"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.16.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BlackBoxOptim]]
deps = ["CPUTime", "Compat", "Distributed", "Distributions", "JSON", "LinearAlgebra", "Printf", "Random", "Requires", "SpatialIndexing", "StatsBase"]
git-tree-sha1 = "9c203a2515b5eeab8f2987614d2b1db83ef03542"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.3"
weakdeps = ["HTTP", "Sockets"]

    [deps.BlackBoxOptim.extensions]
    BlackBoxOptimRealtimePlotServerExt = ["HTTP", "Sockets"]

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.Clapeyron]]
deps = ["BlackBoxOptim", "CSV", "DiffResults", "Downloads", "FillArrays", "ForwardDiff", "JSON3", "LinearAlgebra", "LogExpFunctions", "NLSolvers", "OrderedCollections", "PackedVectorsOfVectors", "PositiveFactorizations", "PrecompileTools", "Preferences", "Roots", "Scratch", "SparseArrays", "StableTasks", "StaticArrays", "Tables", "UUIDs"]
git-tree-sha1 = "85d6593828ba8f7951ef9d3355cf2e5883927340"
uuid = "7c7805af-46cc-48c9-995b-ed0ed2dc909a"
version = "0.6.3"

    [deps.Clapeyron.extensions]
    ClapeyronCoolPropExt = "CoolProp"
    ClapeyronJutulDarcyExt = "JutulDarcy"
    ClapeyronMetaheuristicsExt = "Metaheuristics"
    ClapeyronMultiComponentFlashExt = "MultiComponentFlash"
    ClapeyronSuperancillaries = "EoSSuperancillaries"
    ClapeyronSymbolicsExt = "Symbolics"
    ClapeyronUnitfulExt = "Unitful"

    [deps.Clapeyron.weakdeps]
    CoolProp = "e084ae63-2819-5025-826e-f8e611a84251"
    EoSSuperancillaries = "c1bf003f-4e47-49d9-bdfd-5a4051db3c04"
    JutulDarcy = "82210473-ab04-4dce-b31b-11573c4f8e0a"
    Metaheuristics = "bcdb8e00-2c21-11e9-3065-2b553b22f898"
    MultiComponentFlash = "35e5bd01-9722-4017-9deb-64a5d32478ff"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "13951eb68769ad1cd460cdb2e64e5e95f1bf123d"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d7477ecdafb813ddee2ae727afa94e9dcb5f3fb0"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.112"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "b10bdafd1647f57ace3885143936749d61638c3b"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "7c4195be1649ae622304031ed46a2f4df989f1eb"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.24"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NLSolvers]]
deps = ["IterativeSolvers", "LinearAlgebra", "PositiveFactorizations", "Printf", "Statistics"]
git-tree-sha1 = "78a8c11e01b159a546495dae68aa7f787547b75f"
uuid = "337daf1e-9722-11e9-073e-8b9effe078ba"
version = "0.5.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

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
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PackedVectorsOfVectors]]
git-tree-sha1 = "78a46960967e9e37f81dbf7f61b45b0159637afe"
uuid = "7713531c-48ef-4bdd-9821-9ff7a8736089"
version = "0.1.2"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "650a022b2ce86c7dcfbdecf00f78afeeb20e5655"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3a7c7e5c3f015415637f5debdf8a674aa2c979c4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShortCodes]]
deps = ["Base64", "CodecZlib", "Downloads", "JSON3", "Memoize", "URIs", "UUIDs"]
git-tree-sha1 = "5844ee60d9fd30a891d48bab77ac9e16791a0a57"
uuid = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
version = "0.3.6"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpatialIndexing]]
git-tree-sha1 = "84efe17c77e1f2156a7a0d8a7c163c1e1c7bdaed"
uuid = "d4ead438-fe20-5cc5-a293-4fd39a41b74c"
version = "0.1.6"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StableTasks]]
git-tree-sha1 = "073d5c20d44129b20fe954720b97069579fa403b"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
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
# ╠═b13b2c8c-6366-43e9-936b-6deb67be6db1
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
