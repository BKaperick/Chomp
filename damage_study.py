import math

def fact(n):
	factors = []
	for i in range(2, int(n / 2) + 1):
		for j in range(2, int(n / 2) + 1):
			if i*j == n:
				factors.append(i)
	return len(factors)

def dam(n):
	term1 = (Lmod(n)**2 + Lmod(n) + Lpmod(n)**2 + Lpmod(n)) / 2
	term2 = 0
	if L(n) >= 5:
		for i in range(4,L(n)):
			term2 += fact(i) * i
	print("\tterm1:",term1)
	print("\t\tLmod(n) =",Lmod(n))
	print("\t\tLmod(n-1) =",Lpmod(n))
	print("\tterm2:",term2)
	return int(term1 + term2 - 1)

def dammod(n):
	term1 = (Lmod(n)**2 + Lmod(n) + Lmod(n-1)**2 + Lmod(n-1)) / 2
	term2 = 0
	if L(n) >= 5:
		for i in range(4,L(n)):
			term2 += fact(i) * i
	print("\tterm1:",term1)
	print("\t\tLmod(n) =",Lmod(n))
	print("\t\tLmod(n-1) =",Lpmod(n))
	print("\tterm2:",term2)
	return int(term1 + term2 - 1)

def L_penalty(n):
	penalty = 0
	if L(n) >= 5:
		for i in range(4,L(n)):
			penalty += fact(i)
	return penalty

def factLT(n):
	output = 0
	ans = math.ceil((n + 1) / 2)
	if ans >= 5:
		for i in range(4,ans):
			output += fact(i)
	return output

def L(n):
	return int(math.ceil((n + 1) / 2))
	    
def Lmod(n):
    return L(n - factLT(n))

def Lpmod(n):
    return L(n - factLT(n) - 1)
	    
def Lp(n):
	return L(n-1)
