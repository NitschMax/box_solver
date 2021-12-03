import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
from math import nan

def main():
	N		= 4
	fock_states	= set_of_fock_states(N)

	index		= 4
	fock_states.print()
	state		= fock_states.get_state(index)
	op		= operator(0, 0, 1)
	print(fock_states.find(state) )
	op.hc().act_on(state)
	print(fock_states.find(state) )

	state.greet()
	op.greet()

class set_of_fock_states:
	def __init__(self, N):
		self.states	= construct_states(N)
		self.dict	= dict(zip([list_to_int(x) for x in self.states], range(len(self.states) ) ) )
		self.len	= len(self.states)
		#print(self.states)

	def get_state(self, index, fac=1):
		return fock_state(self.states[index], fac=fac )

	def print(self):
		print(self.states)

	def find(self, state):
		if state.valid:
			return self.dict.get(list_to_int(state.occ) )
		else:
			return nan

class fock_state:
	def __init__(self, occ=[], fac=1):
		self.occ	= np.array(occ )
		self.fac	= fac
		self.cum_occ	= np.cumsum(self.occ) - self.occ
		self.control()

	def greet(self):
		if self.valid:
			validnes	= 'valid'
		else:
			validnes	= 'unvalid'
		print('Hello World! I am a Fock state. My occupations are {} and my prefactor is {}. My occupations are {}.'.format(self.occ, self.fac, validnes) )

	def __str__(self):
		return "{}*{}".format(self.fac, self.occ )

	def copy(self):
		return fock_state(self.occ.copy(), fac=self.fac)

	def multiply(self, fac):
		self.fac	*= fac

	def set_occ(self, index, bool):
		if bool != 0 and bool != 1:
			print('Choosen occupation not possible!')
		else:
			self.occ[index]	= bool

	def control(self):
		self.cum_occ	= np.cumsum(self.occ) - self.occ
		bool_arr_1	= (0 <= self.occ)
		bool_arr_2	= (1 >= self.occ)
		answer		= bool_arr_1.all() and bool_arr_2.all()
		self.valid	= answer
		return self.valid
		
class maj_operator:
	def __init__(self, index, fac=1, lead=[], coupling=[]):
		self.index	= index
		self.fac	= fac
		self.lead	= lead
		if len(lead) != len(coupling):
			print('The coupling to the leads is not valid. This can case mistakes in the tunneling.')
		self.coupling	= coupling

		if len(self.lead) >= 0:
			self.couples	= True
		else:
			self.couples	= False

	def greet(self):
		print('Hello World! I am {}. I couple to lead nr. {} with t={}'.format(self.__str__(), self.lead, self.coupling) )

	def __str__(self):
		return '{}*gamma_{}'.format(self.fac, self.index)

	def maj_into_occ(self):
		a, b	= index_mapping(self.index)
		if b == 0:
			gamma	= [operator(a, 1, 1*self.fac), operator(a, 0, 1*self.fac) ]
		else:
			gamma	= [operator(a, 1, 1j*self.fac), operator(a, 0, -1j*self.fac) ]
		return gamma

def index_mapping(ind):
	k	= int(np.floor(ind/2) )
	l	= np.mod(ind, 2)
	return k, l
		
class operator:
	def __init__(self, index=0, herm=0, fac=1):
		self.index	= index
		self.herm	= herm
		self.fac	= fac
	
	def __mul__(self, op):
		return [self, op]

	def __str__(self):
		if self.herm == 0:
			dagger	= '-'
		else:
			dagger	= '+'
		return '{}*f{}_{}'.format(self.fac, dagger, self.index)

	def greet(self):
		if self.herm == 0:
			nature	= 'destruction'
		else:
			nature	= 'creation'

		print('Hello World! I am a {} operator. I am acting on the {}th occupation and my prefactor is {}.'.format(nature, self.index, self.fac) )

	def act_on(self, state):
		if self.herm == 1:
			state.fac		*= self.fac*(1-state.occ[self.index] )*(-1)**np.abs(state.cum_occ[self.index] )
			state.occ[self.index]	+= 1
		elif self.herm == 0:
			state.fac		*= self.fac*state.occ[self.index]*(-1)**np.abs(state.cum_occ[self.index] )
			state.occ[self.index]	-= 1
		else:
			print('Invalid operator.')
		state.control()

	def hc(self):
		new_op		= operator(self.index, invert(self.herm), self.fac)
		return new_op


def construct_states(N=2):
	set_of_states	= [] 
	
	numbers_el	= list(range(0,N+1,2) )
	for n in numbers_el:
		state		= np.zeros(N, dtype=np.int8)
		state[:n]	= 1
		permuted_states = np.flip(list(multiset_permutations(state ) ), axis=0)
		set_of_states.append(permuted_states )
		
	numbers_el	= list(range(1,N+1,2) )
	for n in numbers_el:
		state		= np.zeros(N, dtype=np.int8)
		state[:n]	= 1
		permuted_states = np.flip(list(multiset_permutations(state ) ), axis=0)
		set_of_states.append(permuted_states )
	
	set_of_states	= np.concatenate(set_of_states)
	#set_of_states	= np.array([fock_state(x) for x in set_of_states] )
	return set_of_states

def get_index(lst, dict):
	index	= dict.get(list_to_int(lst) )
	return index

def list_to_int(lst):
	return int(''.join(map(str, lst)), 2)

def int_to_list(zahl):
	return split('{0:04b}'.format(zahl) )

def split(word):
	return np.array([int(char) for char in word] )

def invert(occ):
	if occ == 0:
		return 1
	else:
		return 0

if __name__=='__main__':
	main()
