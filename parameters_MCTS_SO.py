
# Parameter File

random_state = 1
seed_value = 12321 # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

num_epochs = 1
coeff = 0
training = 100
total_cost = 784.0
#save_step = 5000

integrated = False

pretrain_model = True
random_model = False
retrain_model = False
fit_model = False

quadratic_strat = True
linear_strat = False
const_strat = False

beg = False
end = True

if end:
	further_name = 'end'
elif beg:
	further_name = 'beg'
else:
	further_name = 'none'

if pretrain_model:
	model_name = 'pretrain'
	retrain_sk = False
	if integrated:
		save_name = 'integrated'
		#purge_step = int(save_step/2.0)
		retrain_step = 27#600 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108
	else:
		save_name = 'standalone'
		purge_step = 6000#int(save_step/2.0)
		retrain_step = 6000#108 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108
	if quadratic_strat:
		cost_name = 'quadratic'
	elif linear_strat:
		cost_name = 'linear'
	elif const_strat:
		cost_name = 'constant'

if random_model:
	model_name = 'random'
	retrain_sk = False
	if integrated:
		save_name = 'integrated'
		#purge_step = int(save_step/2.0)
		retrain_step = 27#600 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108
	else:
		save_name = 'standalone'
		purge_step = 6000#int(save_step/2.0)
		retrain_step = 6000#108 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108

	if quadratic_strat:
		cost_name = 'quadratic'
	elif linear_strat:
		cost_name = 'linear'
	elif const_strat:
		cost_name = 'const'

if retrain_model:
	model_name = 'retrain'
	retrain_sk = True
	#purge_step = int(save_step/2.0)
	retrain_step = 27 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108
	if integrated:
		save_name = 'integrated'
		retrain_integrated = True
		#retrain_sk = False
	else:
		save_name = 'standalone'
		retrain_integrated = False

	if quadratic_strat:
		cost_name = 'quadratic'
	elif linear_strat:
		cost_name = 'linear'
	elif const_strat:
		cost_name = 'constant'

if fit_model:
	model_name = 'fit'
	retrain_sk = False
	#retrain_step = 27
	cost_name = 'fit'
	if integrated:
		save_name = 'integrated'
		#purge_step = int(save_step/2.0)
		retrain_step = 600 #2, 3, 4, 6, 9, 12, 15, 18, 27, 36, 54 ,108
	else:
		save_name = 'standalone'
