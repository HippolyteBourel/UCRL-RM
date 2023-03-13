import numpy as np

Ds4 =  [12.24999992, 44.09978343, 44.09978343, 80.6828951,  44.09778012, 75.87158339]

Ds5 =  [12.24999997, 44.09978343, 44.09978343, 80.6828951,  44.09915605, 75.91377992]

Ds6 =  [12.24999999, 44.09978343, 44.09978343, 80.74384987, 44.09968328, 75.93367433]

Ds8 =  [12.25,       44.09995684, 44.09995684, 80.82740418, 44.09995684, 93.09830042]

Ds10 =  [12.25,        44.09999336 , 44.09999336 , 80.84561166,  44.09999336, 112.69728628]

Ds12 = [ 12.25 ,       44.09999908 , 44.09999908 , 80.84909426 , 44.09999908, 132.29604855]


Dcp = [119.85047795616447, 129.63898420493933, 139.43610637878737, 159.02249119528278, 178.6107575239621, 198.20015423159202]

S = 6
l_N = [4, 5, 6, 8, 10, 12]
base_Q = 16
A = 2

Ksa = np.zeros((S, 2))
for s in range(S):
	if (s == 0):
		Ksa[s, 0] = 6
		Ksa[s, 1] = 6
	else:
		Ksa[s, 0] = 2
		Ksa[s, 1] = 2


i = 0
for Ds in [Ds4, Ds5, Ds6, Ds8, Ds10, Ds12]:
	Q = (base_Q + l_N[i])
# compute res0 = sqrt(OA sum(Ds^2))
	res0 = np.sqrt(S*A*sum([Ds[i]**2 for i in range(S)]))
	# compute res1 = sqrt(c'M)
	res1 = 0
	for s in range(S):
		res1 += Ksa[s, 0] * Ds[s]**2 + Ksa[s, 1] * Ds[s]**2
	res1 = np.sqrt(res1)
	# compute res2 = Dcp sqrt(sum Ksa)
	res2 = 0
	for s in range(S):
		res2 += Ksa[s, 0] + Ksa[s, 1]
	res2 = np.sqrt(Q * res2) * Dcp[i]
	# compute res3 = Dcp sqrt(OAQ)
	res3 = Dcp[i] * S * Q * np.sqrt(A)
	print("For N = ", l_N[i], " sqrt(OA \sum_o Ds^2", res0, " \nsqrt(c\'M) = ", res1, "  \nand Dcp sqrt(sum_qsa Ksa) = ", res2, "\nand Dcp OQ sqrt(A) = ", res3)
	i +=1