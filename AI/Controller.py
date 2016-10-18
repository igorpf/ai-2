from .State import State
from random import randint
import numpy
import math
import random
import datetime, time
import operator
import os

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
		pass
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

	# TODO: funcao que calcula recompensa a cada passo
	# Recebe como o parametro a acao executada, o estado anterior e posterior a execucao dessa acao,
	# o numero de passos desde o inicio do episodio, e um booleano indicando se o episodio acabou apos a execucao da acao.
	# Caso o episodio tenha terminado, o ultimo parametro especifica como ele terminou (IA "won", IA "lost", "draw" ou "collision")
    # Todas essas informacoes podem ser usadas para determinar que recompensa voce quer dar para o agente nessa situacao
	def compute_reward(self, action, prev_state, curr_state, nsteps, isEpisodeOver, howEpisodeEnded):
        # Abaixo, um exemplo de funcao de recompensa super simples (mas provavelmente nao muito efetiva)
		if howEpisodeEnded == "win":
			return 100
		else: return -100


	# TODO: Deve consultar a tabela Q e escolher uma acao de acordo com a politica de exploracao
	# Retorna 1 caso a acao desejada seja direita, 2 caso seja esquerda, 3 caso seja nula, e 4 caso seja atirar
	def take_action(self, state):
		pass


	# TODO: Implementa a regra de atualziacao do Q-Learning.
	# Recebe como o parametro a acao executada, o estado anterior e posterior a execucao dessa acao,
	# a recompensa obtida e um booleano indicando se o episodio acabou apos a execucao da acao
	def updateQ(self, action, prev_state, curr_state, reward, isEpisodeOver):
		pass