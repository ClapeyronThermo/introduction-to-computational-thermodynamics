### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 8c7b3b94-07fa-11ed-3b01-b16ab0f77e08
begin
	using Clapeyron, ForwardDiff, Roots, Optim, LinearAlgebra, PolynomialRoots # Numerical packages
	using LaTeXStrings, Plots, ShortCodes, Printf # Display and plotting
	using HypertextLiteral, PlutoUI
	# using JSON2, Tables,Random # Data handling
	import Clapeyron: vdWModel, RKModel, PRModel, cubic_ab, R̄
	PlutoUI.TableOfContents()
end

# ╔═╡ 2119146e-cb56-4cb2-a4ba-7a359f05ca8d
md"""
# Section 2.3 – Cubic equations of state
Cubic equations of state are by far the most popular equations of state. This can mainly be attributed to their long tenure as the only practical equations of state, but also their simple functional form. We refer to such equations as cubics because they can all be written in the following form

$$V^3+AV^2+BV+C=0.$$

This form has a lot of benefits which we will discuss in part 3 of the course.

## Section 2.3.1 – Van der Waals equation

It is likely that most undergraduates will have encountered the van der Waals equation at some point. However, to give a high-level understanding of this equation, let us start with the ideal gas equation

$$p = \frac{Nk_\mathrm{B}T}{V}.$$

As mentioned previously, the ideal gas equation is based on an assumption that particles are infinitesimally small and experience perfectly elastic collisions. However, for most molecules, this assumption is not valid. Firstly, molecules have volume and thus, take up space in the system, thus reducing the total amount of volume available for other molecules to move around in. The excluded volume of a single particle is typically denoted by the parameter $b$. For $N$ particles, we reduce the available volume by $Nb$

$$p = \frac{Nk_\mathrm{B}T}{V-Nb}.$$

However, molecules also experience attractions between them. The impact on the pressure, which is manifested by the collisions of the molecules with the walls of the box, or container, is two-fold. Firstly, as particles on the edge of our box are about to collide with the box surface, the attraction with molecules in the bulk will reduce their velocity. This will reduce their impact with the surface by an amount proportional to the density of particles in the bulk. Similarly, particles interacting with each other would tend to clump together rather than venture to the boundaries of the box, further reducing the pressure. This too will be proportional to the bulk density of our system. As a result, the net change to the pressure will be proportional to the density squared

$$\Delta p \propto -(N/V)^2=-\rho^2.$$

If we characterise this proportionality by a parameter $a$, we can write out the van der Waals equation as

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2a}{V^2}.$$

More commonly written out in molar form as

$$p = \frac{RT}{v_\mathrm{m}-b} - \frac{a}{v^2_\mathrm{m}}.$$

This was a very high-level description of the van der Waals equation; it is possible to derive it using statistical mechanics although, it is worth pointing out, van der Waals himself did not derive it this way. The equation was only ever meant to be empirical! Nevertheless, the first term can be thought of as the repulsive contribution and the second term can be thought of as the attractive contribution. Visually:
"""

# ╔═╡ 0a86eeb1-b8a5-48f0-aaaa-4719e18a3a32
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/vdW.svg" height="300"></center>""")

# ╔═╡ 51306a06-e837-4e83-a892-a6de8617374d
md"""
Interestingly, for all cubics we will discuss, this visual picture does not change significantly. The parameters $a$ and $b$ can be obtained by constraining the equation such that it must pass through the critical pressure and temperature of a given species

$$a = \frac{27}{64} \frac{(RT_\mathrm{c})^2}{p_\mathrm{c}}$$
$$b = \frac{1}{8} \frac{RT_\mathrm{c}}{p_\mathrm{c}},$$

where the subscripts $\mathrm{c}$ denote the critical properties of a species.

Despite its significant contributions towards the development of equations of state, the van der Waals equation is not suitable for practical applications in the liquid phase as it heavily under-estimates the saturated liquid densities.
"""


# ╔═╡ 75207913-c5ef-40bd-932d-c6955803ef1e
begin
	species="methane"
	model = vdW([species])

	Tc,pc,Vc = crit_pure(model)

	N = 400
	T = range(0.3*Tc,Tc,length=N)

	sat = saturation_pressure.(model,T);
end

# ╔═╡ 25b201de-3358-44a1-bf70-904047404872
Exp_methane = [90.694 0.011696 28.142 0.035534;
95.694 0.021469 27.724 0.036070;
100.69 0.036936 27.297 0.036633;
105.69 0.060146 26.862 0.037228;
110.69 0.093451 26.415 0.037857;
115.69 0.13946 25.956 0.038526;
120.69 0.20101 25.484 0.039240;
125.69 0.28110 24.996 0.040006;
130.69 0.38286 24.490 0.040833;
135.69 0.50955 23.964 0.041730;
140.69 0.66451 23.413 0.042711;
145.69 0.85115 22.834 0.043794;
150.69 1.0730 22.222 0.045001;
155.69 1.3336 21.568 0.046366;
160.69 1.6369 20.862 0.047934;
165.69 1.9866 20.090 0.049776;
170.69 2.3872 19.228 0.052006;
175.69 2.8434 18.237 0.054835;
180.69 3.3610 17.033 0.058709;
185.69 3.9477 15.395 0.064957;
90.694 0.011696 0.015630 63.981;
95.694 0.021469 0.027308 36.619;
100.69 0.036936 0.044909 22.267;
105.69 0.060146 0.070195 14.246;
110.69 0.093451 0.10512 9.5125;
115.69 0.13946 0.15185 6.5856;
120.69 0.20101 0.21271 4.7012;
125.69 0.28110 0.29032 3.4445;
130.69 0.38286 0.38757 2.5801;
135.69 0.50955 0.50783 1.9692;
140.69 0.66451 0.65499 1.5267;
145.69 0.85115 0.83383 1.1993;
150.69 1.0730 1.0503 0.95210;
155.69 1.3336 1.3122 0.76208;
160.69 1.6369 1.6301 0.61346;
165.69 1.9866 2.0193 0.49522;
170.69 2.3872 2.5035 0.39944;
175.69 2.8434 3.1233 0.32018;
180.69 3.3610 3.9602 0.25251;
185.69 3.9477 5.2367 0.19096];

# ╔═╡ 1d8a4153-dfb0-49a4-8a10-e550e162ac47
begin
	psat1 = [sat[i][1] for i ∈ 1:N]
	vl1 = [sat[i][2] for i ∈ 1:N]
	vv1 = [sat[i][3] for i ∈ 1:N]

	plot(1e-3./vl1,T,color=:blue,xlim=(0,30),ylim=(75,200),
		title="Vapour–liquid envelope of methane",
		label="van der Waals",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vv1,T,color=:blue,label="")
	scatter!(Exp_methane[:,3],Exp_methane[:,1],color=:white,edgecolor=:blue,label="Experimental")
end

# ╔═╡ 8527090b-637b-4e77-b2c4-826754ebfba0
md"""
Even in the case of methane, which can almost be considered spherical, the van der Waals equation cannot capture the experimental data. A few modifications made by Clausius and Berthelot did come along, but none ever fully dealt with the main issues of the van der Waals equation.
"""

# ╔═╡ 114acd11-0a26-4388-bb5a-d389495cc0a5
md"""
## Section 2.3.2 – Engineering Cubics
"""

# ╔═╡ c15d67d2-28bb-4fba-9f78-b018e187081d
md"""
Almost 50 years after van der Waals first derived his equation of state, the first true step towards a practically useable equation of state was that developed by Redlich and Kwong

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2a}{V(V+Nb)\sqrt{T}},$$

where the parameters $a$ and $b$ can now be obtained from

$$a = 0.42748 \frac{R^2T_\mathrm{c}^{2.5}}{p_\mathrm{c}}$$
$$b = 0.08664 \frac{RT_\mathrm{c}}{p_\mathrm{c}}.$$

The change made to the second term is purely empirical, with no physical meaning aside from improved modelling of gas fugacities (important in vapour–liquid calculations). In comparison to the van der Waals equation, this equation saw a substantial improvement in modelling the vapour phase (and, somewhat, the liquid phase).
"""

# ╔═╡ f18d246f-e54f-4538-ba49-1ce57a5aea71
begin
	model2 = RK([species])
	model3 = SRK([species])
	model4 = PR([species])

	sat2 = saturation_pressure.(model2,T)

	psat2 = [sat2[i][1] for i ∈ 1:N]
	vl2 = [sat2[i][2] for i ∈ 1:N]
	vv2 = [sat2[i][3] for i ∈ 1:N]

	sat3 = saturation_pressure.(model3,T)

	psat3 = [sat3[i][1] for i ∈ 1:N]
	vl3 = [sat3[i][2] for i ∈ 1:N]
	vv3 = [sat3[i][3] for i ∈ 1:N]

	sat4 = saturation_pressure.(model4,T)

	psat4 = [sat4[i][1] for i ∈ 1:N]
	vl4 = [sat4[i][2] for i ∈ 1:N]
	vv4 = [sat4[i][3] for i ∈ 1:N]

	plot(1e-3./vl1,T,color=:blue,xlim=(1e-2,30),ylim=(100,200),
		title="Vapour–liquid envelope of methane",
		label="van der Waals",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vv1,T,color=:blue,label="")
	plot!(1e-3./vl2,T,color=:red,
		label="Redlich–Kwong")
	plot!(1e-3./vv2,T,color=:red,label="")
	# plot!(1e-3./vl3,T,color=:green,
	# 	label="Soave–Redlich–Kwong")
	# plot!(1e-3./vv3,T,color=:green,label="")
	# plot!(1e-3./vl4,T,color=:purple,
	# 	label="Peng–Robinson")
	# plot!(1e-3./vv4,T,color=:purple,label="")
	scatter!(Exp_methane[:,3],Exp_methane[:,1],color=:white,edgecolor=:blue,label="Experimental")
end

# ╔═╡ 5b8e57fa-58d2-4892-ba95-5647894cf1be
md"""
However, the real "game-changer" for cubic equations of state came when Soave introduced the concept of an $\alpha$ function which modified the Redlich–Kwong equation in the following way

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{V(V+Nb)}.$$

Here, $b$ and $a$ are defined almost the same way, except for $a$

$$a = 0.42748 \frac{(RT_\mathrm{c})^2}{p_\mathrm{c}}$$

and $\alpha(T)$, the $\alpha$ function, given by

$$\alpha(T) = (1+(0.480 + 1.547\omega - 0.176\omega^2)(1-(T/T_\mathrm{c})^{0.5}))^2,$$

where $\omega$ is the acentricity, a species-specific parameter, defined as

$$\omega = -\log{(p_\mathrm{sat}/p_\mathrm{c})}-1\,\,\mathrm{at}\,\,T=0.7T_\mathrm{c}.$$

Interestingly, the acentricity does carry some physical meaning: the more-spherical the species is, the closer its value should be to zero (such as methane or the noble gases).

The idea behind the $\alpha$ function is that, if you can nail down, in ($p,T$) space, both the critical point (which the Redlich–Kwong equation already does) and a second point on the saturation curve around 0.7$T_\mathrm{c}$, characterised by the acentricity, then, ideally, you should be able to capture the entire saturation curve. This is indeed what happens for most species.
"""

# ╔═╡ 57eee2ce-2b25-404a-a5b0-dcbc191d6ee6
begin
plot(T,psat1./1e6,color=:blue,yaxis=:log,xlim=(75,200),ylim=(1e-4,1e1),
		title="Saturation curve of methane",
		label="van der Waals",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:bottomright,
		xlabel=L"T / \mathrm{K}", ylabel=L"p / \mathrm{MPa}")
	plot!(T,psat2./1e6,color=:red,
		label="Redlich–Kwong")
	plot!(T,psat3./1e6,color=:green,
		label="Soave–Redlich–Kwong")
	scatter!(Exp_methane[:,1],Exp_methane[:,2],
		color=:white,edgecolor=:blue,label="Experimental")
end

# ╔═╡ ad5b1611-1ae1-44e1-b5de-f367755a3251
Exp_water = [273.16 0.00061165 55497. 1.8019e-05;
283.16 0.0012290 55489. 1.8022e-05;
293.16 0.0023408 55406. 1.8048e-05;
303.16 0.0042494 55264. 1.8095e-05;
313.16 0.0073889 55074. 1.8157e-05;
323.16 0.012358 54842. 1.8234e-05;
333.16 0.019956 54573. 1.8324e-05;
343.16 0.031214 54272. 1.8426e-05;
353.16 0.047434 53941. 1.8539e-05;
363.16 0.070208 53582. 1.8663e-05;
373.16 0.10145 53196. 1.8798e-05;
383.16 0.14343 52785. 1.8945e-05;
393.16 0.19874 52350. 1.9102e-05;
403.16 0.27036 51891. 1.9271e-05;
413.16 0.36164 51408. 1.9452e-05;
423.16 0.47629 50901. 1.9646e-05;
433.16 0.61839 50371. 1.9853e-05;
443.16 0.79238 49816. 2.0074e-05;
453.16 1.0030 49235. 2.0311e-05;
463.16 1.2555 48629. 2.0564e-05;
473.16 1.5553 47995. 2.0835e-05;
483.16 1.9081 47332. 2.1127e-05;
493.16 2.3200 46639. 2.1441e-05;
503.16 2.7976 45911. 2.1781e-05;
513.16 3.3475 45148. 2.2149e-05;
523.16 3.9768 44345. 2.2551e-05;
533.16 4.6930 43497. 2.2990e-05;
543.16 5.5038 42600. 2.3474e-05;
553.16 6.4176 41646. 2.4012e-05;
563.16 7.4429 40626. 2.4615e-05;
573.16 8.5891 39528. 2.5298e-05;
583.16 9.8664 38337. 2.6085e-05;
593.16 11.286 37028. 2.7007e-05;
603.16 12.860 35567. 2.8116e-05;
613.16 14.603 33895. 2.9503e-05;
623.16 16.531 31899. 3.1349e-05;
633.16 18.668 29283. 3.4150e-05;
643.16 21.046 25052. 3.9918e-05;
273.16 0.00061165 0.26947 3.7110;
283.16 0.0012290 0.52250 1.9139;
293.16 0.0023408 0.96164 1.0399;
303.16 0.0042494 1.6892 0.59199;
313.16 0.0073889 2.8458 0.35139;
323.16 0.012358 4.6175 0.21657;
333.16 0.019956 7.2429 0.13807;
343.16 0.031214 11.019 0.090752;
353.16 0.047434 16.307 0.061322;
363.16 0.070208 23.538 0.042484;
373.16 0.10145 33.215 0.030107;
383.16 0.14343 45.916 0.021779;
393.16 0.19874 62.303 0.016051;
403.16 0.27036 83.119 0.012031;
413.16 0.36164 109.20 0.0091575;
423.16 0.47629 141.48 0.0070684;
433.16 0.61839 180.98 0.0055254;
443.16 0.79238 228.87 0.0043693;
453.16 1.0030 286.42 0.0034914;
463.16 1.2555 355.07 0.0028163;
473.16 1.5553 436.44 0.0022913;
483.16 1.9081 532.34 0.0018785;
493.16 2.3200 644.88 0.0015507;
503.16 2.7976 776.45 0.0012879;
513.16 3.3475 929.88 0.0010754;
523.16 3.9768 1108.5 0.00090210;
533.16 4.6930 1316.4 0.00075963;
543.16 5.5038 1558.6 0.00064161;
553.16 6.4176 1841.2 0.00054312;
563.16 7.4429 2172.5 0.00046030;
573.16 8.5891 2563.1 0.00039015;
583.16 9.8664 3028.0 0.00033025;
593.16 11.286 3588.6 0.00027866;
603.16 12.860 4277.7 0.00023377;
613.16 14.603 5149.9 0.00019418;
623.16 16.531 6307.4 0.00015854;
633.16 18.668 7989.7 0.00012516;
643.16 21.046 11209. 8.9212e-5];

# ╔═╡ 46a7072b-b0b8-4fa8-bae5-a114246b9626
Exp_butane = [134.90 6.6566e-07 12645. 7.9082e-05;
144.90 3.8642e-06 12485. 8.0099e-05;
154.90 1.7413e-05 12324. 8.1140e-05;
164.90 6.3929e-05 12164. 8.2209e-05;
174.90 0.00019842 12003. 8.3310e-05;
184.90 0.00053587 11842. 8.4446e-05;
194.90 0.0012882 11680. 8.5620e-05;
204.90 0.0028072 11516. 8.6837e-05;
214.90 0.0056281 11351. 8.8101e-05;
224.90 0.010507 11183. 8.9419e-05;
234.90 0.018450 11014. 9.0795e-05;
244.90 0.030728 10842. 9.2238e-05;
254.90 0.048878 10666. 9.3755e-05;
264.89 0.074696 10487. 9.5358e-05;
274.89 0.11022 10303. 9.7056e-05;
284.89 0.15770 10115. 9.8866e-05;
294.89 0.21960 9920.4 0.00010080;
304.89 0.29853 9719.3 0.00010289;
314.89 0.39727 9510.3 0.00010515;
324.89 0.51871 9292.2 0.00010762;
334.89 0.66591 9063.2 0.00011034;
344.89 0.84204 8821.1 0.00011336;
354.89 1.0504 8563.1 0.00011678;
364.89 1.2946 8285.3 0.00012070;
374.89 1.5784 7981.7 0.00012529;
384.89 1.9061 7643.5 0.00013083;
394.89 2.2826 7254.8 0.00013784;
404.89 2.7139 6783.8 0.00014741;
414.89 3.2086 6141.3 0.00016283;
424.89 3.7816 4423.2 0.00022608;
134.90 6.6566e-07 0.00059350 1684.9;
144.90 3.8642e-06 0.0032076 311.76;
154.90 1.7413e-05 0.013522 73.956;
164.90 6.3929e-05 0.046636 21.443;
174.90 0.00019842 0.13650 7.3259;
184.90 0.00053587 0.34886 2.8665;
194.90 0.0012882 0.79625 1.2559;
204.90 0.0028072 1.6527 0.60507;
214.90 0.0056281 3.1658 0.31588;
224.90 0.010507 5.6645 0.17654;
234.90 0.018450 9.5630 0.10457;
244.90 0.030728 15.361 0.065100;
254.90 0.048878 23.643 0.042295;
264.89 0.074696 35.079 0.028507;
274.89 0.11022 50.426 0.019831;
284.89 0.15770 70.535 0.014177;
294.89 0.21960 96.370 0.010377;
304.89 0.29853 129.03 0.0077501;
314.89 0.39727 169.79 0.0058896;
324.89 0.51871 220.16 0.0045422;
334.89 0.66591 281.98 0.0035464;
344.89 0.84204 357.57 0.0027967;
354.89 1.0504 449.98 0.0022223;
364.89 1.2946 563.39 0.0017750;
374.89 1.5784 703.88 0.0014207;
384.89 1.9061 880.88 0.0011352;
394.89 2.2826 1110.6 0.00090045;
404.89 2.7139 1425.1 0.00070169;
414.89 3.2086 1910.1 0.00052352;
424.89 3.7816 3417.9 0.00029258];

# ╔═╡ 6415202d-6516-438c-8010-b315a035f242
md"""
However, even in cases like _n_-butane, the SRK equation isn't able to model the liquid densities accurately.
"""

# ╔═╡ 497f19b4-e8a4-4770-bdc9-f5fcfee917a2
begin
	mb1 = SRK(["butane"])
	Tcb,pcb,vcb = crit_pure(mb1)

	Tb = range(70,Tcb,length=N)

	satb1 = saturation_pressure.(mb1,Tb)

	psatb1 = [satb1[i][1] for i ∈ 1:N]
	vlb1 = [satb1[i][2] for i ∈ 1:N]
	vvb1 = [satb1[i][3] for i ∈ 1:N]

	plot(1e-3./vl3,T,color=:blue,ylim=(70,440),
		title="Vapour–liquid envelope of methane and n-butane\n using SRK",
		label="methane",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vv3,T,color=:blue,label="")
	plot!(1e-3./vlb1,Tb,color=:red,
		label="n-butane")
	plot!(1e-3./vvb1,Tb,color=:red,label="")
	scatter!(Exp_methane[:,3],Exp_methane[:,1],color=:white,edgecolor=:blue,label="")
	scatter!(Exp_butane[:,3]*1e-3,Exp_butane[:,1],color=:white,edgecolor=:blue,label="")
end


# ╔═╡ 7dc41002-c95c-47a8-8718-e41fd05b7eaf
md"""
It is for this reason that Peng and Robinson (PR) developed their own cubic equation of state, also using their own $\alpha$ function

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{V^2+2NbV+(Nb)^2},$$

where

$$a = 0.45724 \frac{(RT_\mathrm{c})^2}{p_\mathrm{c}}$$
$$b = 0.0778 \frac{RT_\mathrm{c}}{p_\mathrm{c}}$$

and $\alpha(T)$

$$\alpha(T) = (1+(0.37464 + 1.54226\omega - 0.26992\omega^2)(1-(T/T_\mathrm{c})^{0.5}))^2.$$

Indeed, using the PR equation, the improvement in modelling of liquid densities is significant.
"""

# ╔═╡ d5c9c5fb-3e62-4acc-ad2a-f4c692316418
begin
	mb2 = PR(["butane"])

	satb2 = saturation_pressure.(mb2,Tb)

	psatb2= [satb2[i][1] for i ∈ 1:N]
	vlb2 = [satb2[i][2] for i ∈ 1:N]
	vvb2 = [satb2[i][3] for i ∈ 1:N]

	plot(1e-3./vlb1,Tb,color=:blue,ylim=(70,440),
		title="Vapour–liquid envelope of n-butane",
		label="SRK",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vvb1,Tb,color=:blue,label="")
	plot!(1e-3./vlb2,Tb,color=:red,
		label="PR")
	plot!(1e-3./vvb2,Tb,color=:red,label="")
	# plot!(1e-3./vl3,T,color=:green,
	# 	label="Soave–Redlich–Kwong")
	# plot!(1e-3./vv3,T,color=:green,label="")
	# plot!(1e-3./vl4,T,color=:purple,
	# 	label="Peng–Robinson")
	# plot!(1e-3./vv4,T,color=:purple,label="")
	scatter!(Exp_butane[:,3]*1e-3,Exp_butane[:,1],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ 59a53741-74c5-46f0-81b7-2310078e6a3f
md"""
That isn't to say SRK is no longer useful. In fact, both SRK and PR represent the industry standards for equation of state modelling as, depending on what you are trying to model (do fugacities or densities matter more?), one may be more accurate than the other.

However, it is also important to bear in mind what systems these equations of state are intended for: hydrocarbon/natural gases. If you were to try and model something like water using either of these equations, the results would be disappointing.
"""

# ╔═╡ 9ee5e353-e028-4c34-8ea4-73e846c410b7
begin
	mw1 = SRK(["water"])
	mw2 = PR(["water"])
	Tcw,pcw,vcw = crit_pure(mw1)

	Tw = range(270,Tcw,length=N)

	satw1 = saturation_pressure.(mw1,Tw)

	psatw1 = [satw1[i][1] for i ∈ 1:N]
	vlw1 = [satw1[i][2] for i ∈ 1:N]
	vvw1 = [satw1[i][3] for i ∈ 1:N]

	satw2 = saturation_pressure.(mw2,Tw)

	psatw2= [satw2[i][1] for i ∈ 1:N]
	vlw2 = [satw2[i][2] for i ∈ 1:N]
	vvw2 = [satw2[i][3] for i ∈ 1:N]

	plot(1e-3./vlw1,Tw,color=:blue,ylim=(270,660),
		title="Vapour–liquid envelope of water",
		label="SRK",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vvw1,Tw,color=:blue,label="")
	plot!(1e-3./vlw2,Tw,color=:red,
		label="PR")
	plot!(1e-3./vvw2,Tw,color=:red,label="")
	scatter!(Exp_water[:,3]*1e-3,Exp_water[:,1],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ dfab5341-cb23-4878-a868-8a3432677f27
md"""
Nevertheless, for what most systems engineers are interested in, SRK and PR provide an easy way to access the full range of thermodynamic properties we might need. There have been further developments in cubic equation of state modelling (some of which will be highlighted below), including the introduction of a third (e.g., Patel–Teja) and sometimes fourth (e.g., GEOS) parameter to model a wider range of species. However, all retain the simple cubic form, which can be generalised as

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{(V+r_1Nb)(V+r_2Nb)},$$

where

| Equation | $r_1$        | $r_2$        |
|----------|--------------|--------------|
| vdW      | 0            | 0            |
| RK/SRK   | 0            | 1            |
| PR       | $1+\sqrt{2}$ | $1-\sqrt{2}$ |

Meaning, if we can just write a single function to obtain the pressure in terms of the parameters $r_1$ and $r_2$, it would be compatible with all cubic equations of state we may wish to use. However, if we also want to be able to obtain other properties of interest (such as heat capacities, Joule–Thomson coefficients, etc.), it is more convenient to express our equation in terms of the Helmholtz free energy. This can be obtained by integrating the pressure in terms of the volume

$$A = -\int p dV = -Nk_\mathrm{B}T\log{(V-Nb)}-\frac{Na}{b}\frac{\log(V+r_2Nb)-\log(V+r_1Nb)}{r_2-r_1}+c(N,T).$$

The problem here is that we re-introduce the integration constant $c(N,T)$ which, from the previous section, we know arises from the missing ideal contributions. As we know how to handle the ideal term separately, it is easier for us to focus on the residual Helmholtz free energy instead of the total. (This is something we will continue to do in future sections.) To obtain the residual, we deduct the ideal contribution from the total

$$A_\mathrm{res.} = A - A_\mathrm{ideal} = -Nk_\mathrm{B}T\log{(1-Nb/V)}-\frac{Naα(T)}{b}\frac{\log(V+r_2Nb)-\log(V+r_1Nb)}{r_2-r_1}.$$

With this equation, we should be able to obtain any thermodynamic property for any cubic equation of state. For this reason, it will be useful for the reader to implement it themselves.
"""

# ╔═╡ eab4a0fa-b366-4c2e-a6cf-859a2966cbc9
md"""### Task: Implementing a generalised function for cubics"""

# ╔═╡ 75975127-b9c7-4959-a644-ac88e1bcb402
md"""
In this exercise, we assume that we have specified a composition (moles), volume (m³), and temperature (K) of the system and already have the parameters needed to characterise our system ($a$ and $b$). To write a generalised approach for cubic equations of state, we are going to use Julia's multiple dispatch feature where we will write one function `a_res(model::CubicModel,V,T,z)` to obtain the _reduced_ Helmholtz free energy ($a_\mathrm{res.}= A_\mathrm{res.}/(Nk_\mathrm{B}T)$) for _any_ cubic and three functions to give us $r_1$ and $r_2$, (e.g., `cubic_r(model::vdWModel)=(0.,0.)`), which will be used within `a_res`, for _each_ cubic.

As a first step, let us write these `cubic_r` functions.
"""

# ╔═╡ 81ff3c9b-a23a-4466-adcd-f6ce7545ed66
begin
	cubic_r(model::vdWModel) = (0.,0.) # van der Waals
	cubic_r(model::RKModel) = (0.,0.) # Redlich–Kwong and Soave–Redlich–Kwong
	cubic_r(model::PRModel) = (0.,0.) # Peng–Robinson
end

# ╔═╡ de07e69d-d670-4684-bdfd-8570b00403f5
md"""
Before moving on to writing the `a_res` function, there are a few key details to remember.

Firstly, as we are going to be using automatic differentiation, we need to make sure all of our variables are explicitly defined. This may seem obvious since variables like the volume and temperature are quite clearly laid out. However, for equilibrium calculations, where we require the chemical potential, the composition becomes very important. Often, in literature, as one mole is usually assumed, authors often forget to write out explicitly the composition dependence. We have ensured that, wherever a composition dependence is present, it has been written out explicitly.

For consistency, we will assume $a$ and $b$ are in molar units, meaning our generalised equation becomes

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-nb/V)}-\frac{na}{b}\frac{\log(V+r_2nb)-\log(V+r_1nb)}{r_2-r_1},$$

where, for multi-component systems,

$$n = \sum_i z_i.$$

The above is true when implementing any equation of state.

Secondly, in deriving our generalised equation, we neglected one case: where $r_1=r_2=0$ for vdW. Although there are some cubics where $r_1=r_2\neq 0$, for now, we will only consider vdW. Integrating again gives

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-nb/V)}-\frac{n^2a}{V}.$$

With all this in mind, we are ready to implement our own generalised cubic equation of state. (Remember, we are trying to obtain the reduced, residual Helmholtz free energy.)
"""

# ╔═╡ 675b7f18-2b90-4fac-8b0c-9e252f3c923d
function a_res(model::CubicModel,V,T,z)
	n = sum(z)

	# cubic_ab will obtain a and b for any cubic equation of state
	aα,b,c = cubic_ab(model, V, T, z) # ignore c for now

	r1,r2 = cubic_r(model)

	a1 = 0

	if r1==r2
		a2 = 0
	else
		a2 = 0
	end
	return a1+a2
end

# ╔═╡ 8db10907-1f79-48c0-97c6-4ac0b3bd62ab
md"""
The above is a good example of the benefits of multiple dispatch in Julia. We wrote one function which should take in any cubic model (or inputs of type `CubicModel`) and only defined the parameters which are equation-specific. This means that, if we wanted to implement another cubic equation of state in the future, all we would need to add is a new `cubic_r` function.

Nevertheless, with this equation defined, we will be able to obtain any properties of interest! How we do this is the topic of the next part of the course.
"""

# ╔═╡ 3d93db93-97c7-4bb5-85a1-8965f40f601a
md"""
## Section 2.3.3 – $\alpha$ functions
"""

# ╔═╡ 91386684-ea1d-4a56-8437-fb72c45c02b0
md"""
Now that we have an understanding of the advantages and disadvantages of the SRK and PR equation of state, let us go about fixing some of their failings. As we've already mentioned, the $\alpha$ function, introduced by Soave, greatly improved the ability to model the saturation curve of hydrocarbon/natural gas species. However, as we showed with water, this improvement is not universal. To remedy this, numerous authors have developed new $\alpha$ functions. To name a few and the reason for their existence.
* PR-78: Just two years after they initially published their equation of state Peng and Robinson re-parameterised their $\alpha$ function. This version is more accurate than the original for a greater range of species.
* Boston–Matthias: Below the critical point, this $\alpha$ function is the same as the standard SRK and PR $\alpha$ function. The change Boston and Matthias made was for how the $\alpha$ function behaves above the critical point. We will illustrate the impact of this shortly.
* Magoulas–Tassios: A re-parameterised version of the $\alpha$ function using more parameters, intended to be used with a modified PR equation.
* Twu _et al._: An $\alpha$ function with species-specific parameters. As the $\alpha$ function is now fitted for a species directly, this is by far the most accurate $\alpha$ function available. Intended to be used with PR.

As an example, let us compare the default PR and Twu $\alpha$ functions (feel free to switch out the $\alpha$ function).
"""

# ╔═╡ 11b92e61-5b0a-4790-b2db-5d9b3836a0a3
mw = PR(["water"];alpha=TwuAlpha);

# ╔═╡ 86310bfe-d8ad-45f4-8b5f-677642e4c5c4
begin
	satw3 = saturation_pressure.(mw,Tw)

	psatw3= [satw3[i][1] for i ∈ 1:N]
	vlw3 = [satw3[i][2] for i ∈ 1:N]
	vvw3 = [satw3[i][3] for i ∈ 1:N]

	plot(Tw,psatw2./1e6,color=:blue,xlim=(270,400),ylim=(2e-4,5e-1),yaxis=:log,
		title="Vapour–liquid envelope of water",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topleft,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"T / \mathrm{K}")
	plot!(Tw,psatw3./1e6,color=:red,
		label="PR+Twu")
	scatter!(Exp_water[:,1],Exp_water[:,2],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ 473e023d-20a9-4cd8-961b-f6fb17002203
md"""
The impact of the $\alpha$ function is a bit more subtle than just the saturation pressure. It can also have a large impact of vapour–liquid equilibrium properties of mixtures. Although we haven't covered how mixtures are handled using cubics, for the time being, we will only look at the impact of the $\alpha$ function from a high-level perspective.
"""

# ╔═╡ faacd2b7-0218-4848-85d4-6441ec30d155
begin
	mix_pr1 = PR(["benzene","methanol"])
	mix_pr2 = PR(["benzene","methanol"];alpha=TwuAlpha)

	T_mix_pr = 300.15

	x = range(0.,1.,length=N)
	X = Clapeyron.Fractions.FractionVector.(x)

	bub_pr1 = bubble_pressure.(mix_pr1,T_mix_pr,X)
	bub_pr2 = bubble_pressure.(mix_pr2,T_mix_pr,X)

	p_pr1 = [bub_pr1[i][1] for i ∈ 1:N]
	y_pr1 = [bub_pr1[i][4][1] for i ∈ 1:N]

	p_pr2 = [bub_pr2[i][1] for i ∈ 1:N]
	y_pr2 = [bub_pr2[i][4][1] for i ∈ 1:N]

	plot(x,p_pr1./1e6,color=:blue,xlim=(0,1),
		title="pxy diagram of benzene+methanol",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y_pr1,p_pr1./1e6,color=:blue,
		label="")
	plot!(x,p_pr2./1e6,color=:red,
		label="PR+Twu")
	plot!(y_pr2,p_pr2./1e6,color=:red,
		label="")
	annotate!(0.02, 0.0145, text("T=300.15 K", :black, :left, 14))
end


# ╔═╡ 1b34fd27-47ba-4545-aed2-3615cb792886
md"""
Without considering experimental data, it is clear that the $\alpha$ function can have a profound impact on mixture properties. The logic behind this is simple: the saturation pressure of the pure components is the end points for VLE envelope of the mixtures. If you can't get those right, then you shouldn't expect to get the points in-between correctly.

However, the impact can go beyond this. Even when one of the components is supercritical, the $\alpha$ function selected still matters. Consider a mixture of carbon dioxide and carbon monoxide at a temperature where carbon monoxide is supercritical.
"""

# ╔═╡ 67255639-cb5e-414f-82bd-145b0047c76e
begin
	mix_pr3 = PR(["carbon monoxide","carbon dioxide"])
	mix_pr4 = PR(["carbon monoxide","carbon dioxide"];alpha=BMAlpha)

	T_mix_pr1 = 270.15

	x1 = range(0.,0.24,length=N)
	X1 = Clapeyron.Fractions.FractionVector.(x1)
	x2 = range(0.,0.24,length=N)
	X2 = Clapeyron.Fractions.FractionVector.(x2)

	bub_pr3 = bubble_pressure.(mix_pr3,T_mix_pr1,X1)
	bub_pr4 = bubble_pressure.(mix_pr4,T_mix_pr1,X2)

	p_pr3 = [bub_pr3[i][1] for i ∈ 1:N]
	y_pr3 = [bub_pr3[i][4][1] for i ∈ 1:N]

	p_pr4 = [bub_pr4[i][1] for i ∈ 1:N]
	y_pr4 = [bub_pr4[i][4][1] for i ∈ 1:N]

	plot(x1,p_pr3./1e6,color=:blue,xlim=(0,0.5),ylim=(2,12),
		title="pxy diagram of carbon monoxide+carbon dioxide",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:bottomright,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{CO}),y(\mathrm{CO})")
	plot!(y_pr3,p_pr3./1e6,color=:blue,
		label="")
	plot!(x2,p_pr4./1e6,color=:red,
		label="PR+BM")
	plot!(y_pr4,p_pr4./1e6,color=:red,
		label="")
		annotate!(0.02, 11., text("T=270.15 K", :black, :left, 14))

end

# ╔═╡ a768aeed-693a-4f21-8a31-9101cfd3fe68
md"""
The reason for this, and why the Boston–Matthias $\alpha$ function should be used whenever one component is supercritical, is that, the default $\alpha$ functions in PR and SRK both behave unphysically at temperatures above the critical point ($T>3T_\mathrm{c}$), which is a problem when mixtures contain species, like carbon monoxide, which have low critical points.

Overall, when considering which cubic equation of state to use, which $\alpha$ function to use is an important question to ask. For most hydrocarbon systems, the standard SRK and PR equations should be sufficient. However, for more complex systems, it is worth considering not only the pure saturation curves, but the mixed systems as well. In general, the safest would be to use species-specific $\alpha$ functions like the one developed by Twu _et al._ although the parameters may not be available for every species.
"""

# ╔═╡ 46dbdb64-819c-4e5b-a4ee-2b66b571e7aa
md"""
## Section 2.3.4 – Volume translation
"""

# ╔═╡ ba6362ec-3c1a-43c1-8651-637d352bbc1a
md"""
We previously spent a lot of time considering the impact of the $\alpha$ function on the saturation curve of our species. However, what about the liquid densities? Something we know cubics struggle with for complex species like water. In short, nothing much changes.
"""

# ╔═╡ 47cadf7f-0d2f-4fa0-8d2b-7480d2b9517c
begin
	plot(1e-3./vlw2,Tw,color=:blue,ylim=(270,660),
		title="Vapour–liquid envelope of water",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vvw2,Tw,color=:blue,label="")
	plot!(1e-3./vlw3,Tw,color=:red,
		label="PR+Twu")
	plot!(1e-3./vvw3,Tw,color=:red,label="")
	scatter!(Exp_water[:,3]*1e-3,Exp_water[:,1],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ 47d1995c-896e-44c9-b786-e6f6ff182cbf
md"""
This is not ideal as, in some scenarios, we need accurate liquid densities. One very simple correction that has a minimal impact on our original equation is to introduce a volume translation. In this case, the "true" volume and the volume fed into our equation is shifted slightly

$$V_\mathrm{eos} = V - Nc,$$

where $c$ is our shift. The benefit of this approach is that it will not impact our saturation curve as $c$ is just a constant; it will only improve our predictions for the liquid densities. For example, for PR, the Rackett equation provides the shift (for SRK, the Péneloux equation can be used).
"""

# ╔═╡ 4e0ffadd-c762-4444-a87c-d812fb31d37f
mw4 = PR(["water"];alpha=TwuAlpha,
				   translation=RackettTranslation);

# ╔═╡ bb028481-f5e1-48f2-8f2d-1703230ff92a
begin
	satw4 = saturation_pressure.(mw4,Tw)

	psatw4= [satw4[i][1] for i ∈ 1:N]
	vlw4 = [satw4[i][2] for i ∈ 1:N]
	vvw4 = [satw4[i][3] for i ∈ 1:N]

	plot(1e-3./vlw2,Tw,color=:blue,ylim=(270,660),
		title="Vapour–liquid envelope of water",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vvw2,Tw,color=:blue,label="")
	plot!(1e-3./vlw3,Tw,color=:red,
		label="PR+Twu")
	plot!(1e-3./vvw3,Tw,color=:red,label="")
	plot!(1e-3./vlw4,Tw,color=:green,
		label="PR+Twu+Rackett")
	plot!(1e-3./vvw4,Tw,color=:green,label="")
	scatter!(Exp_water[:,3]*1e-3,Exp_water[:,1],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ 54cc3ea9-a512-4344-8ea5-212f4ccc5a59
md"""
Although still not quite ideal for water, it is an improvement over the untranslated results.

One thing to consider is the impact on our generalised equation for the cubics. Introducing the shift gives us a slightly different equation

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-n(c-b)/V)}-\frac{na}{b}\frac{\log(V+n(c+r_2b))-\log(V+n(c+r_1b))}{r_2-r_1}.$$
"""

# ╔═╡ e0593d86-a7d8-41cb-a879-90a9720591a9
md"""
Nevertheless, using the code we previously wrote, incorporating this shift should be quite straightforward.

Generally, volume translations should only be used when we need accurate volumetric properties. If this is not the case, then one can afford to ignore the translation.
"""

# ╔═╡ ab519b39-d547-4bd8-aead-050bdf5a5c9d
md"""
## Section 2.3.5 – Mixing rules
"""

# ╔═╡ c9cc4916-a0c4-498b-bbd3-5d2605f1b382
md"""
Now that we have established all the tools needed to model pure systems using cubics, we now need to consider extending them to model mixtures. Typically, we want a set of $a$ and $b$ parameters that characterise the mixture (we will denote these "one-fluid mixture" parameters as $\bar{a}$ and $\bar{b}$).
"""

# ╔═╡ c971e885-6887-4f38-a907-e7cdb1e52032
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/Mixing.svg" height="400"></center>""")

# ╔═╡ 93a18e36-7769-441c-abfb-235b33035991
md"""
How can we do this? The critical points for the mixtures are generally not known, thus, using the usual definition for $a$ and $b$ is not an option.
"""

# ╔═╡ d71826d0-1b95-4b21-9c5b-125e0c10f9b0
md"""
### Van der Waals one-fluid mixing rule
The simplest and most widely used approximation for obtaining $\bar{a}$ and $\bar{b}$ is a van der Waals one-fluid mixing rule

$$\bar{a}=\sum_i\sum_j x_ix_j a_{ij}$$
$$\bar{b}=\sum_i\sum_j x_ix_j b_{ij},$$

where, if $i=j$, then $a_{ii}$ and $b_{ii}$ are the usual parameters for a pure fluid $i$. If $i\neq j$, then $a_{ij}$ and $b_{ij}$ can be obtained using

$$a_{ij} = \sqrt{a_{ii}a_{jj}}(1-k_{ij})$$
$$b_{ij} = \frac{b_{ii}+b_{jj}}{2}(1-l_{ij}).$$

The above are known as combining rules, and $k_{ij}$ and $l_{ij}$ are known as binary interaction parameters. Typically, the van der Waals mixing rule will work well for mixtures of similar species (e.g. ethane+propane) but struggle with associating (e.g. water+ethanol) and/or size asymmetric (e.g. carbon dioxide+_n_-decane) mixtures. $k_{ij}$ and $l_{ij}$ are fitted using experimental data for mixtures containing species $i$ and $j$ to account for these "non-ideal" interactions. Generally, $l_{ij}=0$, meaning the mixing rule for $\bar{b}$ simplifies to

$$\bar{b}=\sum_ix_i b_{ii}$$

and, in some cases, $k_{ij}$ is assigned a temperature dependence (typically $k_{ij}=k_{ij}^{(0)}+k_{ij}^{(1)}T$). We can see the impact of this parameter below.
"""

# ╔═╡ 22e47090-60de-4ca5-844c-fb45558636a1
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

# ╔═╡ bdcba181-1905-4583-bd2c-444e76933db3
begin
	mix_bm1 = PR(["benzene","methanol"];alpha=TwuAlpha)
	mix_bm2 = PR(["benzene","methanol"];alpha=TwuAlpha,userlocations=["assets/"])

	T_mix_bm = 433.15

	bub_bm1 = bubble_pressure.(mix_bm1,T_mix_bm,X)
	bub_bm2 = bubble_pressure.(mix_bm2,T_mix_bm,X)

	p_bm1 = [bub_bm1[i][1] for i ∈ 1:N]
	y_bm1 = [bub_bm1[i][4][1] for i ∈ 1:N]

	p_bm2 = [bub_bm2[i][1] for i ∈ 1:N]
	y_bm2 = [bub_bm2[i][4][1] for i ∈ 1:N]

	plot(x,p_bm1./1e6,color=:blue,xlim=(0,1),
		title="pxy diagram of benzene+methanol",
		label="kᵢⱼ=0",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y_bm1,p_bm1./1e6,color=:blue,
		label="")
	plot!(x,p_bm2./1e6,color=:red,
		label="kᵢⱼ=0.125")
	plot!(y_bm2,p_bm2./1e6,color=:red,
		label="")
	scatter!(1 .-Exp_MeB[:,1],Exp_MeB[:,3].*0.00689476,label="Experimental",color=:white,edgecolor=:blue)
	scatter!(1 .-Exp_MeB[:,2],Exp_MeB[:,3].*0.00689476,label="",color=:white,edgecolor=:blue)
	annotate!(0.02, 0.75, text("T=433.15 K", :black, :left, 14))
end

# ╔═╡ a4b3d1ca-4251-4f24-aed2-f3c82680918e
md"""
Clearly, this binary interaction parameter can have a profound effect on the predicted equilibria for the mixture. We can see above that the equilibrium goes from being almost ideal to having an azeotrope, as well as agreeing more quantitatively with the experimental data.

There are other mixing rules similar to the van der Waals one-fluid mixing rule (e.g. Kay's rule, Rao's rule, etc.). However, all of these require binary interaction parameters to accurately model mixtures. These binary interactions can usually be found in literature, although tools like ASPEN and gPROMS have large databases available. If these are not available, it is recommended to simply fit these parameters using any available experimental data.

Naturally, there comes the limitation that we sometimes need to model systems for which there are no binary interaction parameters or experimental data available in literature.
"""

# ╔═╡ 967d43b8-deb2-4365-98d3-65d798395e76
md"""
### EoS/$G^E$ mixing rules
Having now seen the limitations of simple mixing rules in cubics, we now consider another class of mixing rules. In the previous section we showed how effective activity coefficient-based models were for modelling equilibrium properties of mixture systems, despite being unable to model pure systems and limited to a few properties. What if we could "borrow" this better modelling from the activity coefficient models, and use it together with cubics?

The basic ideal behind $G^E$ mixing rules is, we set the excess Gibbs free energy obtained from the cubic equation of state to that obtained from activity models

$$g^E_\mathrm{cubic}(T,p,z)=g^E_\mathrm{act.}(T,z).$$

The difficulty is that activity coefficient models are pressure-independent. Thus, at which pressure do we set this equality? This depends on which mixing rule we use. The first such mixing rule derived was by Huron and Vidal, who took the infinite pressure limit, giving the mixing rule

$$\frac{\bar{a}}{\bar{b}}=\sum_ix_i\frac{a_i}{b_i}-\frac{G^E}{\lambda},$$

where $G^E$ is obtained from the activity-coefficient model and $\lambda$ is specific to the equation of state. Taking the opposite limit of zero pressure, the mixing rules of Michelsen and, Wong and Sandler are other alternatives (which are too complex to write here). The interesting aspect here is that there is no restriction as to which activity coefficient model can be used here (Wilson, NRTL, UNIQUAC, etc.). For example (feel free to switch out the mixing rule and activity model):
"""

# ╔═╡ 1de4e49b-36cd-4140-8219-c197f5e42898
mix = PR(["benzene","methanol"];alpha=TwuAlpha,mixing=HVRule,activity=Wilson);

# ╔═╡ 64e96e8f-0d13-46bc-93c8-fe293807fe04
begin
	mix_bm3 = PR(["benzene","methanol"];alpha=TwuAlpha,mixing=MHV2Rule,activity=UNIFAC)
	bub_bm  = bubble_pressure.(mix,T_mix_bm,X)
	bub_bm3 = bubble_pressure.(mix_bm3,T_mix_bm,X)

	p_bm = [bub_bm[i][1] for i ∈ 1:N]
	y_bm = [bub_bm[i][4][1] for i ∈ 1:N]

	p_bm3 = [bub_bm3[i][1] for i ∈ 1:N]
	y_bm3 = [bub_bm3[i][4][1] for i ∈ 1:N]

	plot(x,p_bm1./1e6,color=:blue,xlim=(0,1),
		title="pxy diagram of benzene+methanol",
		label="PR",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:bottomleft,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y_bm1,p_bm1./1e6,color=:blue,
		label="")
	plot!(x,p_bm./1e6,color=:red,
		label="PR+HV+Wilson")
	plot!(y_bm,p_bm./1e6,color=:red,
		label="")
	plot!(x,p_bm3./1e6,color=:green,
		label="PR+MHV2+UNIFAC")
	plot!(y_bm3,p_bm3./1e6,color=:green,
		label="")
	scatter!(1 .-Exp_MeB[:,1],Exp_MeB[:,3].*0.00689476,label="Experimental",color=:white,edgecolor=:blue)
	scatter!(1 .-Exp_MeB[:,2],Exp_MeB[:,3].*0.00689476,label="",color=:white,edgecolor=:blue)
	annotate!(0.75, 1.8, text("T=433.15 K", :black, :left, 14))
end

# ╔═╡ bf4d6a0f-3c32-4379-8acc-d724923d728e
md"""
As we can see, the results obtained from these mixing rules are substantially better than those obtained using the simple van der Waals one-fluid mixing rule. However, the above also illustrates the big advantage of such a mixing rule. While Wilson, NRTL and UNIQUAC are all species-specific methods, models like UNIFAC are group-contribution based. This means, as long as the groups for a species are available, we will be able to use this mixing rule to _predict_ the mixture phase equilibria!

In terms of recommendations, if possible, it is always best to validate these mixing rules against experimental data. If binary interaction parameters have been fitted against experimental data, it is usually easier and more trustworthy to use the simpler mixing rules. However, if one must use a $G^E$ mixing rule, generally speaking, the Michelsen, and Wong–Sandler mixing rules are typically the most reliable, coupled with any of the activity-coefficient models (naturally, if one can use species-specific approaches, that is preferred).
"""

# ╔═╡ e31698b2-90bf-4c77-a207-867f9e5e8e33
md"""
## Section 2.3.6 – Predictive Cubics
Now that we've gone through the different types of cubics, $\alpha$ functions, volume translation and mixing rules, it is time to bring them together.
"""

# ╔═╡ bdcc5469-1876-4cd4-9df1-aa660cc8abd0
m = PR(["benzene","methanol"];alpha=TwuAlpha,translation=RackettTranslation,
							  mixing=HVRule,activity=UNIFAC);

# ╔═╡ 80caf67b-b4ae-4cb8-873d-f5ebc9814e30
md"""
However, there are two cubics that make use of all the above methods to provide some of the most accurate equations of state available.
* Predictive SRK: Combines the standard Soave $\alpha$ function, Péneloux volume translation, the Michelsen first order mixing rule and its own version of UNIFAC.
* Volume-translated PR: Combines Twu _et al._'s $\alpha$ function, Rackett volume translation, a modified Huron–Vidal mixing rule and its own version of UNIFAC.
These equations of state are, for most mixtures of interest in industry, almost as accurate as the high-accuracy empirical models (GERG-2008). Both of these approaches are predictive for mixtures, only requiring the critical temperature and pressure to be used. If one can use either of these methods, they are highly recommended. Try to create your own cubic equation of state to see if it rivals their accuracy!
"""

# ╔═╡ 156de269-904c-4312-937e-600c07d088a1
begin
	mix_bm5 = PSRK(["benzene","methanol"])

	bub     = bubble_pressure.(m,T_mix_bm,X)
	bub_bm5 = bubble_pressure.(mix_bm5,T_mix_bm,X)

	pbub  = [bub[i][1] for i ∈ 1:N]
	y = [bub[i][4][1] for i ∈ 1:N]
	p_bm5 = [bub_bm5[i][1] for i ∈ 1:N]
	y_bm5 = [bub_bm5[i][4][1] for i ∈ 1:N]

	plot(x,pbub./1e6,color=:blue,xlim=(0,1),
		title="pxy diagram of benzene+methanol",
		label="Your model",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box,
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:bottomleft,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y,pbub./1e6,color=:blue,
		label="")
	plot!(x,p_bm5./1e6,color=:red,
		label="PSRK")
	plot!(y_bm5,p_bm5./1e6,color=:red,
		label="")
	scatter!(1 .-Exp_MeB[:,1],Exp_MeB[:,3].*0.00689476,label="Experimental",color=:white,edgecolor=:blue)
	scatter!(1 .-Exp_MeB[:,2],Exp_MeB[:,3].*0.00689476,label="",color=:white,edgecolor=:blue)
		annotate!(0.75, 1.9, text("T=433.15 K", :black, :left, 14))
end

# ╔═╡ 73a7e252-8ce8-4640-afea-f7abe28249d2
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ 5d29af5b-b3df-4f77-8ed8-a8970fa8b066
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ 5718c170-7e95-4615-bce6-99b102e9dcbd
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 958b06ac-5863-4e99-b704-75113abe2fa2
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

# ╔═╡ 0149b559-4957-48b4-949f-b9aab7643f28
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ 89b0587e-6fc8-4aec-b2ff-b77e459e7db1
let
a_vdw = a_res(model,1e-3,298.15,[1.])
a_vdw2 = a_res(model,1e-3,298.15,[2.])

a_rk = a_res(mb1,1e-3,298.15,[1.])
a_pr = a_res(mb2,1e-3,298.15,[1.])
a_pr2 = a_res(mb2,1e-3,298.15,[2.])

a_vdw_sol = Clapeyron.a_res(model,1e-3,298.15,[1.])
a_vdw2_sol = Clapeyron.a_res(model,1e-3,298.15,[2.])

a_rk_sol = Clapeyron.a_res(mb1,1e-3,298.15,[1.])
a_pr_sol = Clapeyron.a_res(mb2,1e-3,298.15,[1.])
a_pr2_sol = Clapeyron.a_res(mb2,1e-3,298.15,[2.])
	if (a_vdw≈a_vdw_sol)
		if (a_rk≈a_rk_sol) && (a_pr≈a_pr_sol)
			if (a_pr2≈a_pr2_sol) && (a_vdw2≈a_vdw2_sol)
					correct()
			elseif (a_pr2≈a_pr2_sol) && !(a_vdw2≈a_vdw2_sol)
					almost(md"Make sure the composition dependence is correctly written for vdW")
			elseif !(a_pr2≈a_pr2_sol) && (a_vdw2≈a_vdw2_sol)
					almost(md"Make sure the composition dependence is correctly written for PR and SRK")
				else
					still_missing(md"The general equations are correct but the composition dependence is missing")
				end
		elseif (a_rk≈a_rk_sol) && !(a_pr≈a_pr_sol)
				almost(md"Make sure `cubic_r` is defined correctly for PR")
		elseif !(a_rk≈a_rk_sol) && (a_pr≈a_pr_sol)
				almost(md"Make sure `cubic_r` is defined correctly for SRK")
			else
				almost(md"Make sure either `cubic_r` is defined correctly for SRK and PR, or that the second term for these equations is correct")
			end
		else
			if (a_rk≈a_rk_sol) && (a_pr≈a_pr_sol)
				almost(md"Make sure the second term is defined correctly for vdW")
			elseif (a_rk≈a_rk_sol) && !(a_pr≈a_pr_sol)
				almost(md"Make sure `cubic_r` is defined correctly for PR and the second term is defined correctly for vdW")
			elseif !(a_rk≈a_rk_sol) && (a_pr≈a_pr_sol)
				almost(md"Make sure `cubic_r` is defined correctly for SRK and the second term is defined correctly for vdW")
			else
				almost(md"Something is missing... perhaps the first term is defined incorrectly?")
			end
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
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "6e3fa7fa6f1092cbb75af6ad510dfbce587a6e08"

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
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

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
# ╟─8c7b3b94-07fa-11ed-3b01-b16ab0f77e08
# ╟─2119146e-cb56-4cb2-a4ba-7a359f05ca8d
# ╟─0a86eeb1-b8a5-48f0-aaaa-4719e18a3a32
# ╟─51306a06-e837-4e83-a892-a6de8617374d
# ╠═75207913-c5ef-40bd-932d-c6955803ef1e
# ╟─25b201de-3358-44a1-bf70-904047404872
# ╟─1d8a4153-dfb0-49a4-8a10-e550e162ac47
# ╟─8527090b-637b-4e77-b2c4-826754ebfba0
# ╟─114acd11-0a26-4388-bb5a-d389495cc0a5
# ╟─c15d67d2-28bb-4fba-9f78-b018e187081d
# ╟─f18d246f-e54f-4538-ba49-1ce57a5aea71
# ╟─5b8e57fa-58d2-4892-ba95-5647894cf1be
# ╟─57eee2ce-2b25-404a-a5b0-dcbc191d6ee6
# ╟─ad5b1611-1ae1-44e1-b5de-f367755a3251
# ╟─46a7072b-b0b8-4fa8-bae5-a114246b9626
# ╟─6415202d-6516-438c-8010-b315a035f242
# ╟─497f19b4-e8a4-4770-bdc9-f5fcfee917a2
# ╟─7dc41002-c95c-47a8-8718-e41fd05b7eaf
# ╟─d5c9c5fb-3e62-4acc-ad2a-f4c692316418
# ╟─59a53741-74c5-46f0-81b7-2310078e6a3f
# ╟─9ee5e353-e028-4c34-8ea4-73e846c410b7
# ╟─dfab5341-cb23-4878-a868-8a3432677f27
# ╟─eab4a0fa-b366-4c2e-a6cf-859a2966cbc9
# ╟─75975127-b9c7-4959-a644-ac88e1bcb402
# ╠═81ff3c9b-a23a-4466-adcd-f6ce7545ed66
# ╟─de07e69d-d670-4684-bdfd-8570b00403f5
# ╠═675b7f18-2b90-4fac-8b0c-9e252f3c923d
# ╟─89b0587e-6fc8-4aec-b2ff-b77e459e7db1
# ╟─8db10907-1f79-48c0-97c6-4ac0b3bd62ab
# ╟─3d93db93-97c7-4bb5-85a1-8965f40f601a
# ╟─91386684-ea1d-4a56-8437-fb72c45c02b0
# ╠═11b92e61-5b0a-4790-b2db-5d9b3836a0a3
# ╟─86310bfe-d8ad-45f4-8b5f-677642e4c5c4
# ╟─473e023d-20a9-4cd8-961b-f6fb17002203
# ╟─faacd2b7-0218-4848-85d4-6441ec30d155
# ╟─1b34fd27-47ba-4545-aed2-3615cb792886
# ╟─67255639-cb5e-414f-82bd-145b0047c76e
# ╟─a768aeed-693a-4f21-8a31-9101cfd3fe68
# ╟─46dbdb64-819c-4e5b-a4ee-2b66b571e7aa
# ╟─ba6362ec-3c1a-43c1-8651-637d352bbc1a
# ╟─47cadf7f-0d2f-4fa0-8d2b-7480d2b9517c
# ╟─47d1995c-896e-44c9-b786-e6f6ff182cbf
# ╠═4e0ffadd-c762-4444-a87c-d812fb31d37f
# ╟─bb028481-f5e1-48f2-8f2d-1703230ff92a
# ╟─54cc3ea9-a512-4344-8ea5-212f4ccc5a59
# ╟─e0593d86-a7d8-41cb-a879-90a9720591a9
# ╟─ab519b39-d547-4bd8-aead-050bdf5a5c9d
# ╟─c9cc4916-a0c4-498b-bbd3-5d2605f1b382
# ╟─c971e885-6887-4f38-a907-e7cdb1e52032
# ╟─93a18e36-7769-441c-abfb-235b33035991
# ╟─d71826d0-1b95-4b21-9c5b-125e0c10f9b0
# ╟─22e47090-60de-4ca5-844c-fb45558636a1
# ╟─bdcba181-1905-4583-bd2c-444e76933db3
# ╟─a4b3d1ca-4251-4f24-aed2-f3c82680918e
# ╟─967d43b8-deb2-4365-98d3-65d798395e76
# ╠═1de4e49b-36cd-4140-8219-c197f5e42898
# ╟─64e96e8f-0d13-46bc-93c8-fe293807fe04
# ╟─bf4d6a0f-3c32-4379-8acc-d724923d728e
# ╟─e31698b2-90bf-4c77-a207-867f9e5e8e33
# ╠═bdcc5469-1876-4cd4-9df1-aa660cc8abd0
# ╟─80caf67b-b4ae-4cb8-873d-f5ebc9814e30
# ╟─156de269-904c-4312-937e-600c07d088a1
# ╟─73a7e252-8ce8-4640-afea-f7abe28249d2
# ╟─5d29af5b-b3df-4f77-8ed8-a8970fa8b066
# ╟─0149b559-4957-48b4-949f-b9aab7643f28
# ╟─5718c170-7e95-4615-bce6-99b102e9dcbd
# ╟─958b06ac-5863-4e99-b704-75113abe2fa2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
