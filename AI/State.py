class State:
	def __init__(self,dist_enemy, enemy_sight, dist_arrow, arrow_sight):		
        # grava no estado o valor dos sensores do robo. Esses sensores serao utilizados no calculo de features (em compute_features)
		self.dist_enemy = dist_enemy 
		self.enemy_sight = enemy_sight
		self.dist_arrow = dist_arrow 
		self.arrow_sight = arrow_sight


    # TODO: calcula features, com base nos valores dos sensores. 
	def compute_features(self):
        # Abaixo, usamos como features apenas os sensores propriamente ditos. Os grupos podem aumentar
        # o conjunto de features para fornecer ao agente mais informacoes relevantes à escolha de acoes
		return [self.dist_enemy, self.enemy_sight, self.dist_arrow, self.arrow_sight]

	# TODO: funcao responsavel por discretizar as features calculadas por compute_features(). O numero de niveis
    # a ser usado na discretizacao de cada feature fica a criterio de cada grupo. Mais niveis implica
    # mais estados (e portanto aprendizado mais lento), mas tambem prove ao agente informacao mais detalhada
    # a respeito do que esta ocorrendo---p.ex., a respeito da distancia exata do inimigo, etc. Os niveis
    # de discretizacao utilizados precisam ser informados pela funcao discretization_levels
	def discretize_features(self, features):
		pass

	# TODO: deve retornar um vetor onde o i-ésimo elemento é o numero de niveis usados pra discretizar a i-ésima feature calculada por compute_features
	def discretization_levels(self):
		pass

# retorna o estado propriamente dito do sistema---isto é, um vetor discreto de features, para utilizacao no algoritmo Q-Learning
	def get_state(self):
		features = self.compute_features()
		features = self.discretize_features(features)
		return features

    # calcula um identificador unico para cada estado do sistema (isto é, mapeia o conjunto de features discretizas de estado para um valor inteiro unico)
	def get_state_id(self, features): 
		primos=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193]
	   
		terms = [primos[i]**features[i] for i in range(len(features))]

		s_id=1
		for i in range(len(features)):
			s_id = s_id * terms[i]

		return s_id

    # retorna o numero total de estados possiveis no sistemas. Isso dependera de quantas features voce criou em compute_features,
    # e de que forma decidiu discretiza-las, em discretize_features (i.e., quantos niveis de discretizacao foram utilizados)
	def get_n_states(self):
		v = self.discretization_levels()
		num = 1

		for i in (v):
			num *= i

		return num

    # retorna uma lista com todos os possiveis estados nos quais o sistema pode se encontrar
	def states_list(self):
		list = []

		v = self.discretization_levels()

		featureRanges = []
		for i in range(len(v)):
			featureRanges.append(range(0,v[i]))

		for i in range(len(featureRanges)):
			featureRanges[i] = [[j] for j in featureRanges[i]]

		list = featureRanges.pop(0)

		while len(featureRanges)>0:
			nextFeature = featureRanges.pop(0)
			list = [i+j for i in list for j in nextFeature]

		return list
		    