# Parameter File for MCTS

random_state = 1
seed_value = 12321

num_epochs = 1
training = 100

save_step = 10000

integrated = False

pretrain_model = True
random_model = False
retrain_model = False

classifier_name = 'lr'

if pretrain_model:
	model_name = ''
	retrain_sk = False
	retrain_step = int(save_step/2.0)
	if integrated:
		save_name = 'integrated'
		retrain_step_pol = int(save_step/2.0)
	else:
		save_name = 'standalone'

if random_model:
	model_name = 'random'
	retrain_sk = False
	retrain_step = int(save_step/2.0)
	if integrated:
		save_name = 'integrated'
		retrain_step_pol = int(save_step/2.0)
	else:
		save_name = 'standalone'

if retrain_model:
	model_name = ''
	retrain_sk = True
	purge_step = save_step
	retrain_step = int(save_step/2.0)
	if integrated:
		save_name = 'integrated'
		retrain_step_pol = int(save_step/2.0)
	else:
		save_name = 'standalone'