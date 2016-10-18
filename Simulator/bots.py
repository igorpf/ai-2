import AI
from random import randint

class Bots:
	
	#difficulty:
	#lazy_bot (sempre girando) -> 0
	#random_bot (comandos aleatorios) -> 1
	#normal_bot (mistura random e ninja) -> 2
	#ninja_bot (aimbot que sabe desviar) -> 3

	def __init__(self, difficulty):
		self.difficulty = difficulty
		self.last_command = 5
		self.command_counter = 0
		self.last_distance = 0
		self.ninja_dodge = False

	def take_random_action(self):
		if self.command_counter < 30: #to keep bot from being too crazy -> changes action after 1 second
			self.command_counter += 1
			return self.last_command
		else:
			rand = randint(0,4)
			self.last_command = rand
			self.command_counter = 0
			return rand

	def take_ninja_action(self,state):
		if state.arrow_sight == True or state.dist_arrow < 2:
			return 1 #1 - direita - try to evade
		elif state.enemy_sight == True:
			return 4 #4 - atira - try to kill
		else: 
			return 2 #2 - esquerda - try to find the player	
		

	def take_action(self,state):
		if self.difficulty == 0: #lazy_bot
			return 2 #2 - esquerda

		elif self.difficulty == 1: #random_bot
			return self.take_random_action()

		elif self.difficulty == 2: #normal_bot
			rand = randint(0,9)
			if rand < 5:
				return self.take_random_action()
			else:
				return self.take_ninja_action(state)
		else: #ninja_bot
			return self.take_ninja_action(state)
