### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 8c7b3b94-07fa-11ed-3b01-b16ab0f77e08
begin
	using Clapeyron, ForwardDiff, Roots, Optim, LinearAlgebra, PolynomialRoots # Numerical packages
	using LaTeXStrings, Plots, ShortCodes, Printf # Display and plotting
	using HypertextLiteral
	# using JSON2, Tables,Random # Data handling
	import Clapeyron: vdWModel, RKModel, PRModel, cubic_ab, R̄
end

# ╔═╡ 2119146e-cb56-4cb2-a4ba-7a359f05ca8d
md"""
### Section 2.3 
# Cubic equations of state
Cubic equations of state are by far the most-popular equations of state. This can mainly be attributed to the their long tenure as the only practical equations of state, but also their simple functional form. We refer to such equations as cubics because they can all be written in the following form:

$$V^3+AV^2+BV+C=0$$

This form has a lot of benefits which we will discuss in part 3 of the course.

### Section 2.3.1
## Van der Waals equation

It is likely that most undergraduates will have encountered the van der Waals equation at some point. However, to give a high-level understanding of this equation, let us start with the ideal gas equation:

$$p = \frac{Nk_\mathrm{B}T}{V}$$

As mentioned previously, the ideal gas equation assumes that particles are infinitesmally small and experience perfectly elastic collisions. However, for most molecules, this is not the case. Firstly, molecules have volume and thus, take up space in the system, thus reducing the total amount of volume available for species to move around in. The volume of a single particle is typically denoted by the parameter $b$. For $N$ particles, we reduce the available volume by $Nb$:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb}$$

However, molecules also experience attractions between them. The impact on the pressure is two-fold. Firstly, as particles on the edge of our box are about to collide with the box surface, the attraction with molecules in the bulk will reduce their velocity. This will be proportional to the density of particles in the bulk. Similarly, particles interacting together would much rather clump together rather than venture out to the boundaries of the box, thus further reducing the pressure. This too will be proportional to the bulk density of our system. As a result, the net change to the pressure will be proportional to the density squared:

$$\Delta p \propto -\rho^2$$

If we characterise this proportionality by a parameter $a$, we can write out the van der Waals equation as:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2a}{V^2}$$

More commonly written out in molar form as:

$$p = \frac{RT}{v_m-b} - \frac{a}{v^2_m}$$

This was a very high-level description of the van der Waals equation; it is possible to derive it using statistical mechanics although, it is worth pointing out, van der Waals himself did not derive it this way. The equation was only ever meant to be empirical! Nevertheless, the first term can be thought of as the repulsive contribution and the second term can be thought of as the attractive contribution. Visually:
"""

# ╔═╡ 0a86eeb1-b8a5-48f0-aaaa-4719e18a3a32
@htl("""<center><img src="https://github.com/lucpaoli/introduction-to-computational-thermodynamics/raw/main/Part%202%20-%20Equations%20of%20State/assets/vdW.svg" height="300"></center>""")

# ╔═╡ 51306a06-e837-4e83-a892-a6de8617374d
md"""
Interestingly, for all cubics we will discuss, this visual picture does not change singificantly. The parameters $a_m$ and $b_m$ can be obtained by constraining the equation such that it must pass through the critical point of a given species:

$$a = \frac{27}{64} \frac{(RT_c)^2}{p_c}$$
$$b = \frac{1}{8} \frac{RT_c}{p_c}$$

where the subscripts $c$ denote the critical properties of a species.

Despite its significant contributions towards the development of equations of state, the van der Waals equation is not suitable for practical applications in the liquid phase as it heavily under-estimates the saturated liquid densities:
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
		title="Vapour-liquid envelope of methane",
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
Even in the case of methane, which can almost be considered spherical, the van der Waals cannot reproduce the experimental data. A few modifications made by Clausius and Berthelot did come along, but none ever fully dealt with the main issues of the van der Waals equation.
"""

# ╔═╡ 114acd11-0a26-4388-bb5a-d389495cc0a5
md"""
### Section 2.3.2
## Engineering Cubics
"""

# ╔═╡ c15d67d2-28bb-4fba-9f78-b018e187081d
md"""
Almost 50 years after van der Waals first derived his equation of state, the first true step towards a practically useable equation of state was that developed by Redlich and Kwong:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2a}{V(V+Nb)\sqrt{T}}$$

where the parameters $a$ and $b$ can now be obtained from:

$$a = 0.42748 \frac{R^2T_c^{2.5}}{p_c}$$
$$b = 0.08664 \frac{RT_c}{p_c}$$

The change made to the second term is purely empirical, with no physical meaning aside from improved modelling of gas fugacities (important in vapour-liquid calculations). In comparison to the van der Waals equation, this equation saw a substantial improvement in modelling the vapour phase:
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
		title="Vapour-liquid envelope of methane",
		label="van der Waals",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"T / \mathrm{K}", xlabel=L"\rho / (\mathrm{mol}/\mathrm{L})")
	plot!(1e-3./vv1,T,color=:blue,label="")
	plot!(1e-3./vl2,T,color=:red,
		label="Redlich-Kwong")
	plot!(1e-3./vv2,T,color=:red,label="")
	# plot!(1e-3./vl3,T,color=:green,
	# 	label="Soave-Redlich-Kwong")
	# plot!(1e-3./vv3,T,color=:green,label="")
	# plot!(1e-3./vl4,T,color=:purple,
	# 	label="Peng-Robinson")
	# plot!(1e-3./vv4,T,color=:purple,label="")
	scatter!(Exp_methane[:,3],Exp_methane[:,1],color=:white,edgecolor=:blue,label="Experimental")
end

# ╔═╡ 5b8e57fa-58d2-4892-ba95-5647894cf1be
md"""
However, the real 'game-changer' for cubic equations of state came when Soave introduced the concept of an $\alpha$-function which modified the Redlich—Kwong equation in the following way:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{V(V+Nb)}\,,$$
Here, $b$ and $a$ are defined almost the same way, except for $a_0$:

$$a = 0.42748 \frac{(RT_c)^2}{p_c}$$

and $\alpha(T)$, the $\alpha$-function, given by:

$$\alpha(T) = (1+(0.480 + 1.547\omega - 0.176\omega^2)(1-(T/T_c))^{0.5})^2$$

where $\omega$ is the acentricity, a species-specific parameter, defined as:

$$\omega = -\log{(p_\mathrm{sat}/p_c)}-1\,\,\mathrm{at}\,\,T=0.7T_c$$

Interestingly, the acentricity does carry some physical meaning: the more-spherical the species is, the closer its value should be to zero (such as methane or the noble gases). 

The idea behind the $\alpha$-function is that, if you can nail-down both the critical point (which the Redlich—Kwong equation already does) and a second point on the saturation curve around 0.7$T_c$, characterised by the acentricity, then, ideally, you should be able to capture the entire saturation curve. This is indeed what happens for most species:
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
		label="Redlich-Kwong")
	plot!(T,psat3./1e6,color=:green,
		label="Soave-Redlich-Kwong")
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
However, even in cases like $n$-butane, the SRK equation isn't able to model the liquid densities accurately:
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
		title="Vapour-liquid envelope of methane and n-butane\n using SRK",
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
It is for this reason that Peng and Robinson (PR) developed their own cubic equation of state, also using their own $\alpha$-function:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{V^2+2NbV+(Nb)^2}\,,$$
where:

$$a_0 = 0.45724 \frac{(RT)^2}{p_c}$$
$$b = 0.0778 \frac{RT_c}{p_c}$$

and $\alpha(T)$:

$$\alpha(T) = (1+(0.37464 + 1.54226\omega - 0.26992\omega^2)(1-(T/T_c))^{0.5})^2$$
Indeed, using the PR equation, the improvement in modelling of liquid densities is significant:
"""

# ╔═╡ d5c9c5fb-3e62-4acc-ad2a-f4c692316418
begin
	mb2 = PR(["butane"])

	satb2 = saturation_pressure.(mb2,Tb)

	psatb2= [satb2[i][1] for i ∈ 1:N]
	vlb2 = [satb2[i][2] for i ∈ 1:N]
	vvb2 = [satb2[i][3] for i ∈ 1:N]

	plot(1e-3./vlb1,Tb,color=:blue,ylim=(70,440),
		title="Vapour-liquid envelope of n-butane",
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
	# 	label="Soave-Redlich-Kwong")
	# plot!(1e-3./vv3,T,color=:green,label="")
	# plot!(1e-3./vl4,T,color=:purple,
	# 	label="Peng-Robinson")
	# plot!(1e-3./vv4,T,color=:purple,label="")
	scatter!(Exp_butane[:,3]*1e-3,Exp_butane[:,1],color=:white,edgecolor=:blue,label="")
end

# ╔═╡ 59a53741-74c5-46f0-81b7-2310078e6a3f
md"""
That isn't to say SRK is no longer useful. In fact, both SRK and PR represent the industry standards for equation of state modelling as, depending on what you are trying to model, one may be more-accurate than the other. 

However, it is also important to bear in mind what systems these equations of state are intended for: hydrocarbon / natural gases. If you were to try and model something like water using either of these equations the results would be disappointing:
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
		title="Vapour-liquid envelope of water",
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
Nevertheless, for most systems engineers are interested in, SRK and PR provide an easy way to access the full range of thermodynamic properties we might need. There have been further developments in cubic equation of state modelling (some of which will be highlighted below), including the introduction of a third (e.g., Patel-Teja) and sometimes fourth (e.g., GEOS) parameter to model a wider range of species. However, all retain the simple cubic form which can be generalised as:

$$p = \frac{Nk_\mathrm{B}T}{V-Nb} - \frac{N^2aα(T)}{(V+r_1Nb)(V+r_2Nb)}\,,$$

where:

| Equation | $r_1$        | $r_2$        |
|----------|--------------|--------------|
| vdW      | 0            | 0            |
| RK/SRK   | 0            | 1            |
| PR       | $1+\sqrt{2}$ | $1-\sqrt{2}$ |

Meaning, if we can just write a single function to obtain the pressure in terms of the parameters $r_1$ and $r_2$, it would be compatible with all cubic equations of state we may which to use. However, if we also want to be able to obtain other properties of interest (such as heat capacities, Joule-Thomson coefficients, etc.), it is more-convenient to express our equation in terms of the Helmholtz free energy. This can be obtained by integrating the pressure in terms of the volume:

$$A = -\int p dV = -Nk_\mathrm{B}T\log{(V-Nb)}-\frac{Na}{b}\frac{\log(V+r_2Nb)-\log(V+r_1Nb)}{r_2-r_1}+c(N,T)$$

The problem here is that we re-introduce the integration constant $c(N,T)$ which, from the previous section, we know arises from the missing ideal contributions. As we know how to handle the ideal term separately, it is easier for us to focus on the residual Helmholtz free energy instead of the total (this is something we will continue to do in future sections). To obtain the residual, we deduct the ideal contribution from the total:

$$A_\mathrm{res.} = A - A_\mathrm{ideal} = -Nk_\mathrm{B}T\log{(1-Nb/V)}-\frac{Naα(T)}{b}\frac{\log(V+r_2Nb)-\log(V+r_1Nb)}{r_2-r_1}$$

With this equation, we should be able to obtain any thermodynamic property for any cubic equation of state. For this reason, it will be useful for the reader to implement it themselves.
	"""

# ╔═╡ eab4a0fa-b366-4c2e-a6cf-859a2966cbc9
md"""### Task: Implementing a generalised function for cubics"""

# ╔═╡ 75975127-b9c7-4959-a644-ac88e1bcb402
md"""
In this exercise, we assume that we have specified a composition (moles), volume (m$^3$), and temperature (K) of the system and already have the parameters needed to characterise our system ($a$ and $b$). To write a generalised approach for cubic equations of state, were are going to use Julia's multiple dispatch feature where we will write one function `a_res(model::CubicModel,V,T,z)` to obtain the _reduced_ Helmholtz free energy ($a_\mathrm{res.}= A_\mathrm{res.}/(Nk_\mathrm{B}T)$) for _any_ cubic and three functions to give us $r_1$ and $r_2$, (e.g., `cubic_r(model::vdWModel)=(0.,0.)`), which will be used within `a_res`, for _each_ cubic.

As a first step, let us write these `cubic_r` functions:
"""

# ╔═╡ 81ff3c9b-a23a-4466-adcd-f6ce7545ed66
begin
	cubic_r(model::vdWModel) = (0.,0.) # van der Waals
	cubic_r(model::RKModel) = (0.,1.) # Redlich-Kwong and Soave-Redlich-Kwong
	cubic_r(model::PRModel) = (1+sqrt(2),1-sqrt(2)) # Peng-Robinson
end

# ╔═╡ de07e69d-d670-4684-bdfd-8570b00403f5
md"""
Before moving on to writing the `a_res` function, there are a few key details to remember:

Firstly, as we are going to be using automatic differentiation, we need to make sure all of our variables are explicitly defined. This may seem obvious since variables like the volume and temperature are quite clearly laid out. However, for equilibrium calculations, where we require the chemical potential, the composition becomes very important. Often, in literature, as one mole is usually assumed, authors often forget to write out explicitly the composition dependence. We have ensured that, wherever a composition dependence is present, it has been written out explicitly. 

For consistency, we will assume $a$ and $b$ are in molar units, meaning our generalised equation becomes:

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-nb/V)}-\frac{na}{b}\frac{\log(V+r_2nb)-\log(V+r_1nb)}{r_2-r_1}$$

where, for multi-component systems:

$$n = \sum_i z_i$$

The above is true when implementing any equation of state.

Secondly, in deriving our generalised equation, we neglected one case: where $r_1=r_2=0$ for vdW. Although there are some cubics where $r_1=r_2\neq 0$, for now, we will only consider vdW. Integrating again gives:

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-nb/V)}-\frac{n^2a}{V}$$

With all this in mind, we are ready to implement our own generalised cubic equation of state (remember, we are trying to obtain the reduced, residual Helmholtz free energy):
"""

# ╔═╡ 675b7f18-2b90-4fac-8b0c-9e252f3c923d
function a_res(model::CubicModel,V,T,z)
	n = sum(z)

	# cubic_ab will obtain a and b for any cubic equation of state
	aα,b,c = cubic_ab(model, V, T, z) # ignore c for now

	r1,r2 = cubic_r(model)
	
	a1 = -log(1-n*b/V)
	
	if r1==r2
		a2 = -n*aα/(V*R̄*T)
	else
		a2 = -aα/(b*R̄*T)*(log(V+r2*n*b)-log(V+r1*n*b))/(r2-r1)
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
### Section 2.3.3
## $\alpha$ functions
"""

# ╔═╡ 91386684-ea1d-4a56-8437-fb72c45c02b0
md"""
Now that we have an understanding of the advantages and disadvantages of the SRK and PR equation of state, let us go about fixing some of their failings. As we've already mentioned, the $\alpha$ function, introduced by Soave, greatly improved the ability to model the saturation curve of hydrocarbon / natural gas species. However, as we showed with water, this improvement is not universal. To remedy this, numerous authors have developed new $\alpha$ functions. To name a few and the reason for their existence:
* PR-78: Just two years after they initially published their equation of state Peng and Robinson re-parameterised their $\alpha$ function. This version is more accurate than the original for a greater range of species.
* Boston-Matthias: Below the critical point, this $\alpha$ function is the same as the standard SRK and PR $\alpha$ function. The change Boston and Matthias made was for how the $\alpha$ function behaves above the critical point. We will illustrate the impact of this shortly.
* Magoulas-Tassios: A re-parameterised version of the $\alpha$ function using more parameters, intended to be used with a modified PR equation.
* Twu _et al._: An $\alpha$ function with species-specific parameters. As the $\alpha$ function is now fitted for a species directly, this is by far the most accurate $\alpha$ function available. Intended to be used with PR.

As an example, let us compare the default PR and Twu $\alpha$-functions (feel free to switch out the $\alpha$ function):
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
		title="Vapour-liquid envelope of water",
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
The impact of the $\alpha$ function is a bit more-subtle than just the saturation pressure. It can also have a large impact of vapour-liquid equilibrium properties of mixtures. Although we haven't covered how mixtures are handled within cubics, for the time being, we will only look at the impact of the $\alpha$-function from a high-level perspect:
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
Without considering experimental data, it is clear that the $\alpha$ function can have a profound impact on mixture properties. The logic behind this is simple: the saturation pressure of the pure components is the end points for VLE envelope of the mixutres. If you can't get those right, then you shouldn't expect to get the points inbetween correctly.

However, the impact can go beyond this. Even when one of the components is supercritical, the $\alpha$ function selected still matters. Consider a mixture of carbon dioxide and carbon monoxide at a temperature where carbon monoxide is supercritical:
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
The reason for this, and why the Boston-Matthias $\alpha$ function should be used whenever one component is supercritical is that, the default $\alpha$ functions in PR and SRK both behave unphysically at temperature above the critical point ($T>3T_c$) which is a problem when mixtures contain species, like carbon monoxide, which have low critical points.

Overall, when considering which cubic equation of state to use, which $\alpha$ function to use is an important question to ask. For most hydrocarbon systems, the standard SRK and PR equations should be sufficient. However, for more-complex systems, it is worth considering not only the pure saturation curves, but the mixed systems as well. In general, the safest would be to use species-specific $\alpha$ functions like the one developed by Twu _et al._ although the parameters may not be available for every species.
"""

# ╔═╡ 46dbdb64-819c-4e5b-a4ee-2b66b571e7aa
md"""
### Section 2.3.4
## Volume translation
"""

# ╔═╡ ba6362ec-3c1a-43c1-8651-637d352bbc1a
md"""
We previously spent a lot of time considering the impact of the $\alpha$ function on the saturation curve of our species. However, what about the liquid densities? Something we know cubics struggle with for complex species like water. In short, nothing much changes:
"""

# ╔═╡ 47cadf7f-0d2f-4fa0-8d2b-7480d2b9517c
begin
	plot(1e-3./vlw2,Tw,color=:blue,ylim=(270,660),
		title="Vapour-liquid envelope of water",
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
This is not ideal as, in some scenarios, we need accurate liquid densities. One very simple correction that has a minimal impact on our original equation is to introduce a volume translation. In this case, The `true` volume and the volume fed into our equation is shift slightly:

$$V_\mathrm{eos} = V - Nc$$

where $c$ is our shift. The benefit of this approach is that it will not impact our saturation curve as $c$ is just a constant; it will only improve our predictions for the liquid densities. For example, for PR, the Rackett equation provides the shift (for SRK, the Péneloux equation can be used):
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
		title="Vapour-liquid envelope of water",
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

One this to consider is the impact on our generalised equation for the cubics. Introducing the shift gives us a slightly different equation:

$$A_\mathrm{res.} = -n\bar{R}T\log{(1-n(c-b)/V)}-\frac{na}{b}\frac{\log(V+n(c+r_2b))-\log(V+n(c+r_1b))}{r_2-r_1}$$
	"""

# ╔═╡ e0593d86-a7d8-41cb-a879-90a9720591a9
md"""
Nevertheless, using the code we previously wrote, incorporating this shift should be quite straightforward.

Generally, volume translations should only be used when we need accurate volumetric properties. If this is not the case, then one can afford to ignore the translation. 
"""

# ╔═╡ ab519b39-d547-4bd8-aead-050bdf5a5c9d
md"""
### Section 2.3.5
## Mixing rules
"""

# ╔═╡ c9cc4916-a0c4-498b-bbd3-5d2605f1b382
md"""
Now that we have established all the tools needed to model pure systems using cubics, we now need to consider extending them to model mixtures. Typically, we want a set of $a$ and $b$ parameters that characterise the mixture (we will denote this `one-fluid mixture' parameters as $\bar{a}$ and $\bar{b}$):
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
The simplest and most-widely used approximation for obtaining $\bar{a}$ and $\bar{b}$ is the van der Waals one-fluid mixing rule:

$$\bar{a}=\sum_i\sum_j x_ix_j a_{ij}$$
$$\bar{b}=\sum_i\sum_j x_ix_j b_{ij}$$

where, if $i=j$, then $a_{ii}$ and $b_{ii}$ are the usual parameters for a pure fluid $i$. If $i\neq j$, then $a_{ij}$ and $b_{ij}$ can be obtained using:

$$a_{ij} = \sqrt{a_{ii}a_{jj}}(1-k_{ij})$$
$$b_{ij} = \frac{b_{ii}+b_{jj}}{2}(1-l_{ij})$$

The above are known as combining rules, and $k_{ij}$ and $l_{ij}$ are known as binary interaction parameters. Typically, the van der Waals mixing rule will work well for mixtures of similar species (e.g. ethane+propane) but struggle with associating (e.g. water+ethanol) and/or size asymmetric (e.g. carbon dioxide+n-decane) mixtures. $k_{ij}$ and $l_{ij}$ are fitted against mixtures containing species $i$ and $j$ to account for these `non-ideal' interactions. Generally, $l_{ij}=0$, meaning the mixing rule for $\bar{b}$ simplifies to: 

$$\bar{b}=\sum_ix_i b_{ii}$$

and, in some case, $k_{ij}$ is assigned a temperature dependence (typically $k_{ij}=k_{ij}^{(0)}+k_{ij}^{(1)}T$). We can see the impact of this parameter below:
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
		label=L"k_{ij}=0",
		yguidefontsize=16, xguidefontsize=16,
		legendfont=font(10), framestyle=:box, 
		tick_direction=:out, grid=:off,foreground_color_legend = nothing,background_color_legend = nothing,legend=:topright,
		ylabel=L"p / \mathrm{MPa}", xlabel=L"x(\mathrm{benzene}),y(\mathrm{benzene})")
	plot!(y_bm1,p_bm1./1e6,color=:blue,
		label="")
	plot!(x,p_bm2./1e6,color=:red,
		label=L"k_{ij}=0.125")
	plot!(y_bm2,p_bm2./1e6,color=:red,
		label="")
	scatter!(1 .-Exp_MeB[:,1],Exp_MeB[:,3].*0.00689476,label=L"\mathrm{Experimental}",color=:white,edgecolor=:blue)
	scatter!(1 .-Exp_MeB[:,2],Exp_MeB[:,3].*0.00689476,label="",color=:white,edgecolor=:blue)
	annotate!(0.02, 0.75, text("T=433.15 K", :black, :left, 14))

end

# ╔═╡ a4b3d1ca-4251-4f24-aed2-f3c82680918e
md"""
Clearly, this binary interaction parameter can have a profound effect on the predicted equilibria for the mixture. We can see above that the equilibria goes from being almost ideal to having an azeotrope, as well as agreeing more-quantitatively with the experimental data.

There are other mixing rules similar to the van der Waals one-fluid mixing rule (e.g. Kay's rule, Rao's rule, etc.). However, all of these require binary interaction parameters to accurately model mixtures. These binary interactions can usually be found in literature, although tools like ASPEN and gPROMS have large databases available. If these are not available, it is recommended to simply fit these parameters to any available experimental data. 

Naturally, there comes the limitation that we sometimes need to model systems for which there are no binary interaction parameters or experimental data available in literature.
"""

# ╔═╡ 967d43b8-deb2-4365-98d3-65d798395e76
md"""
### EoS/$G^E$ mixing rules
Having now seen the limitations of simple mixing rules in cubics, we now consider another class of mixing rules. In the previous section, we showed how effective activity coefficient-based models were for modelling equilibrium properties of mixture systems, despite being unable to model pure systems and limited to a few properties. What if we could 'borrow' this better modelling from the activity coefficent models, and use it within cubics? 

The basic ideal behind $G^E$ mixing rules is we set the excess Gibbs free energy obtained from the cubic equation of state to that obtained from activity models:

$$g^E_\mathrm{cubic}(T,p,z)=g^E_\mathrm{act.}(T,z)$$

The difficulty is that activity coefficient models are pressure-independent. Thus, at which pressure do we set this equality? This depends on which mixing rule we use. The first such mixing rule derived was by Huron and Vidal which took the infinite pressure limit, giving the following mixing rule:

$$\frac{\bar{a}}{\bar{b}}=\sum_ix_i\frac{a_i}{b_i}-\frac{G^E}{\lambda}$$

where $G^E$ is obtained from the activity coefficient model and $\lambda$ is equation of state-specific. Taking the opposite limit of zero pressure, the mixing rules of Michelsen and, Wong and Sandler are other alternatives (which are too complex to write here). The interesting aspect here is that there is no restriction as to which activity coefficient model can be used here (Wilson, NRTL, UNIQUAC, etc.). For example (feel free to switch out the mixing rule and activity model):
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
As we can see, the results obtained from these mixing rules are substantially better than those obtained using the simple van der Waals one-fluid mixing rule. However, the above also illustrates the big advantage of such a mixing rule. While Wilson, NRTL and UNIQUAC are all species-specific method, models like UNIFAC are group-contribution based. This means, as long as the groups for a species are available, we will be able to use this mixing rule to _predict_ the mixture phase equilibria!

In terms of recommendations, if possible, it is always best to validate these mixing rules against experimental data. If binary interaction parameters have been fitted against experimental data, it is usually easier and more-trustworthy to use the simpler mixing rules. However, if one must use a $G^E$ mixing rule, generally speaking, the Michelsen and, Wong-Sandler mixing rules are typically the most-reliable, coupled with any of the activity coefficient models (naturally, if one can use species-specific approaches, that is preferred).
"""

# ╔═╡ e31698b2-90bf-4c77-a207-867f9e5e8e33
md"""
### Section 2.3.6
## Predictive Cubics
Now that we've gone through the different types of cubics, $\alpha$ functions, volume translation and mixing rules, it is time to bring them together:
"""

# ╔═╡ bdcc5469-1876-4cd4-9df1-aa660cc8abd0
m = PR(["benzene","methanol"];alpha=TwuAlpha,translation=RackettTranslation,
							  mixing=HVRule,activity=UNIFAC);

# ╔═╡ 80caf67b-b4ae-4cb8-873d-f5ebc9814e30
md"""
However, there are two cubics that make use of all the above methods to provide some of the most-accurate equations of state available:
* Predictive SRK: Combines the standard Soave $\alpha$ function, Peneloux volume translation, the Michelsen first order mixing rule and its own version of UNIFAC.
* Volume-translated PR: Combines Twu _et al._'s $\alpha$ function, Rackett volume translation, a modified Huron-Vidal mixing rule and its own version of UNIFAC.
These equations of state are, for most mixtures of interest in industry, almost as accurate as the high-accuracy empirical models (GERG-2008). Both of these approaches are predictive for mixtures, only requiring the critical point to be used. If one can use either of these methods, they are highly recommended. Try to create your own cubic equation of state to see if it rivals their accuracy!
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
		title=L"pxy\;\textrm{diagram\; of\;benzene+methanol}",
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
PolynomialRoots = "3a141323-8675-5d76-9d11-e1df1406c778"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"

[compat]
Clapeyron = "~0.3.7"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Optim = "~1.7.0"
Plots = "~1.31.3"
PolynomialRoots = "~1.0.0"
Roots = "~2.0.2"
ShortCodes = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "b86326280b812b4b9c42bf735609317e6b02f964"

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
git-tree-sha1 = "7d255eb1d2e409335835dc8624c35d97453011eb"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.14"

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

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

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
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

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
git-tree-sha1 = "5a1e85f3aed2e0d3d99a4068037c8582597b89cf"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.3"

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
git-tree-sha1 = "472d044a1c8df2b062b23f222573ad6837a615ba"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.19"

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
