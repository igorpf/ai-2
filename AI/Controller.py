# -*- coding: utf-8 -*-
from .State import State
from random import randint
import numpy
import math
import random
import datetime, time
import operator
import os
from itertools import (takewhile,repeat)

#Conta a quantidade de linhas de um arquivo
#retirado de http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
#e adaptado para funcionar em Python 2
def rawincount(filename):
	f = open(filename, 'rb')
	bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
	return sum( buf.count(b'\n') for buf in bufgen )

class Controller:

	def __init__(self, load, state):
		self.state = state
		self.init_table_Q(load, state)


	# TODO: carrega a tabela Q de um arquivo (se load!=None, entao load ira conter o nome do arquivo a ser carregado),
	# ou, caso load==None, a funcao de inicializar uma tabela Q manualmente.
	# Dica: a tabela Q possui um valor para cada possivel par de estado e acao. Cada objeto do tipo State possui um id unico
	# (calculado por State.get_state_id), o qual pode ser usado para indexar a sua tabela Q, juntamente com o indice da acao.
	# Para criacao da tabela Q, pode ser importante saber o numero total de estados do sistema. Isso dependera de quantas features
	# voce utilizar e em quantos niveis ira discretiza-las (ver arquivo State.py para mais detalhes). O numero total de
	# estados do sistema pode ser obtido atraves do metodo State.get_n_states.
	# Uma lista completa com os estados propriamente ditos pode ser obtida atraves do metodo State.states_list.
	def init_table_Q(self,load, state):
		states = state.states_list()
		if load != None:
			if len(states)*4 != rawincount(load):
				print "Número de valores está errado!"
				exit()
			else:
				file = open(load)
		self.table_Q = {}
		for state in states:
			for action in [1,2,3,4]:
				# listas não são hashable, por isso transformo para tupla para usar no dicionário, que possui acesso
				# eficiente
				self.table_Q[tuple(state+[action])] = file.readline() if load !=None else random.random()
		if load != None:
			file.close()

		'''if load == None:
			#ler arquivo para inicializar Q
		else:
			#inicializar manualmente'''

	# TODO: salvar a tabela Q aprendida para um arquivo--para posteriormente poder ser lida por init_table_Q
	def save_table_Q(self, episode, state):
		#Escrever nesse arquivo que esta sendo criado na pasta params
		if episode > 0 and episode % 10 == 0:
			if not os.path.exists("./params"):
				os.makedirs("./params")
			output = open("./params/%s.txt" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S'), "w+")
			for state in sorted(self.table_Q):
				output.write(str(self.table_Q[state])+"\n")

	# TODO: funcao que calcula recompensa a cada passo
	# Recebe como o parametro a acao executada, o estado anterior e posterior a execucao dessa acao,
	# o numero de passos desde o inicio do episodio, e um booleano indicando se o episodio acabou apos a execucao da acao.
	# Caso o episodio tenha terminado, o ultimo parametro especifica como ele terminou (IA "won", IA "lost", "draw" ou "collision")
	# Todas essas informacoes podem ser usadas para determinar que recompensa voce quer dar para o agente nessa situacao
	def compute_reward(self, action, prev_state, curr_state, nsteps, isEpisodeOver, howEpisodeEnded):
		# Igor -> Falta melhorar a recompensa

		#Armando -> Alterei o esquema de recompensa para não ser um valor exato caso a situação x aconteça, mas sim um somatorio de tudo que ocorreu durante a jogada
		reward = 0
		if isEpisodeOver:
			if howEpisodeEnded == "win":
				reward += 5
			elif howEpisodeEnded == "lost" or howEpisodeEnded == "collision":
				reward += -5
			else:
				reward += -1
		else:
			reward += -0.08
			prev_feats = prev_state.discretize_features(prev_state.compute_features())
			curr_feats = curr_state.discretize_features(curr_state.compute_features())

			#Recompensa tiro
			if prev_feats[1] and action == 4: #se o inimigo estava à vista e atirou
				reward += 2
			if not prev_feats[1] and action == 4: #atirou sem ver
				reward += -2

			#Evita colisão
			if curr_feats[0] < 4 & curr_feats[0] < prev_feats[0]: #Caso a distancia do inimigo seja inferior a um limite e o passo resultou em uma arpoximação ainda maior gera uma recompensa negativa
				reward += -3
			elif curr_feats[0] < 4 & curr_feats[0] > prev_feats[0]:	#Caso tenha se afastado enquanto a distancia seja baixa, ganha reforço positivo
				reward += 3

		reward *= 0.99 ** nsteps #diminuindo as recompensas futuras
		return reward

	# TODO: Deve consultar a tabela Q e escolher uma acao de acordo com a politica de exploracao
	# Retorna 1 caso a acao desejada seja direita, 2 caso seja esquerda, 3 caso seja nula, e 4 caso seja atirar
	def take_action(self, state):
		feats = state.discretize_features(state.compute_features())
		keys = [tuple(feats+[act]) for act in [1,2,3,4]]
		look_up_Q = [tuple([key,self.table_Q[key]]) for key in keys]
		best_actions = sorted(look_up_Q, key=(lambda x: x[1]) ,reverse=True)
		if random.random() < 0.1: #experimenta coisas novas em 10% das vezes
			return best_actions[randint(1,3)][0][-1]
		else:
			return best_actions[0][0][-1]

	# TODO: Implementa a regra de atualziacao do Q-Learning.
	# Recebe como o parametro a acao executada, o estado anterior e posterior a execucao dessa acao,
	# a recompensa obtida e um booleano indicando se o episodio acabou apos a execucao da acao
	def updateQ(self, action, prev_state, curr_state, reward, isEpisodeOver):
		alpha = 0.5
		gama = 1
		prev_feats = prev_state.discretize_features(prev_state.compute_features())
		curr_feats = curr_state.discretize_features(curr_state.compute_features())

		keys = [tuple(curr_feats + [act]) for act in [1, 2, 3, 4]]
		look_up_Q = [tuple([key, self.table_Q[key]]) for key in keys]
		best_action = sorted(look_up_Q, key=(lambda x: x[1]), reverse=True)[0][1]

		prev_state_key = tuple(prev_feats+[action])
		self.table_Q[prev_state_key] = (1-alpha)*self.table_Q[prev_state_key] + alpha*(reward + gama * best_action)