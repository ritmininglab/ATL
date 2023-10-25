import numpy as np
import sys
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
import pickle
from query_strategies import ATLSampling
from query_strategies.atTools import *
from risk_estimators import BiasedRiskEstimator, TrueRiskEstimator

if __name__ =='__main__':

	NUM_TR_INIT_LB = int(sys.argv[1])
	NUM_TE_INIT_LB = int(sys.argv[2])
	NUM_TR_QUERY = int(sys.argv[3])
	NUM_TE_QUERY = int(sys.argv[4])
	NUM_ROUND = int(sys.argv[5])
	DATA_NAME = str(sys.argv[6])
	NUM_ROUND_TEST = int(sys.argv[7])
	FILE_NAME = str(sys.argv[8])

	TRIALS=1
	TESTEVERY=2#after every 4 training samples we sample a new test batch
	SEEDs=np.arange(TRIALS)
	#FEEDBACK_PERCENTAGE=0.5#Before convergence, How many test would be used
	CONVERGETHRES=0.03
	convergeIters=[]
	means=[]
	varss=[]
	accs=[]
	hold_accs=[]
	hold_means=[]#test accuracy of the hold out dataset
	hold_vars=[]#test var of the hold out dataset


	args_pool = {'MNIST':
					{'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
					'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
					'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
					'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
					'optimizer_adam_args':{'lr': 0.01}},
				'FashionMNIST':
					{'n_epoch': 50, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
					'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
					'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
					'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
					'optimizer_adam_args':{'lr': 0.01}},
				'SVHN':
					{'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
					'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
					'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
					'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
					'optimizer_adam_args':{'lr': 0.01}},
				'CIFAR10':
					# {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
					{'n_epoch': 50, 'transform': transforms.Compose([transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
					'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
					'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
					'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
					'optimizer_adam_args':{'lr': 0.01}}
				}
	args = args_pool[DATA_NAME]
	if DATA_NAME=='SVHN':
		NumClass = 10
		InDim = 32*32*3
		OutDim = 10
	elif DATA_NAME=='MNIST':
		NumClass = 10
		InDim = 28*28
		OutDim = 10
	elif DATA_NAME=='FashionMNIST':
		NumClass = 10
		InDim = 28*28
		OutDim = 10
	elif DATA_NAME=='CIFAR10':
		NumClass = 10
		InDim = 32*32*3
		OutDim = 10
	
	for FEEDBACK_PERCENTAGE in [0.5]:
		pred_hold_list_all = []
		SEEDs = [11,13,15,17,19]

		R_vt_var_list=[]
		R_final_var_list=[]
		R_fb_var_list=[]
		R_cur_var_list=[]
		Acc_var_list=[]
		Acc_FB_var_list=[]
		R_vtfb_var_list =[]
		rtest_var_list = []
		q_list_all=[]
		for seed in SEEDs:
			CONVERGE=False #
			# set seed
			np.random.seed(seed)
			torch.manual_seed(seed)

			torch.backends.cudnn.enabled = False
			# load dataset
			X_tr, Y_tr, X_hold_te, Y_hold_te = get_dataset(DATA_NAME)
			X_tr = X_tr[:30000]
			Y_tr = Y_tr[:30000]
			print('holdshape',X_hold_te.shape)
			# start experiment
			
			n_pool = len(Y_tr)
			print('number of labeled pool: {}'.format(NUM_TR_INIT_LB))
			print('number of unlabeled pool: {}'.format(n_pool - NUM_TR_INIT_LB - NUM_TE_INIT_LB))
			print('number of testing pool: {}'.format(NUM_TE_INIT_LB))
			
			# generate initial labeled pool
			idxs_tr_lb = np.zeros(n_pool, dtype=bool)
			idxs_te_lb = np.zeros(n_pool, dtype=bool)
			idxs_tmp = np.arange(n_pool) #this is used to randomly ini the train and test index
			np.random.shuffle(idxs_tmp)
		
			trainInd=[]
			testInd=[]
			convergeIter=[]
			rnq_list = []
			rnq_finalList = []
			rtr_list = []
			rtrue_list = []
			feedback_list = []
			rhold_list = []
			R_fold_list = []
			R_cur_list = []
			R_vt_list = []
			R_vtfb_list = []
			rtest_list = []
 
			idxs_tr_lb[idxs_tmp[:NUM_TR_INIT_LB]] = True
			X_te=X_tr[idxs_tr_lb]
			Y_te=Y_tr[idxs_tr_lb]

			
			# load network
			net = get_net(DATA_NAME,modelType='nn',in_dim = InDim,out_dim = OutDim)
			net_test=get_net(DATA_NAME,modelType='nn',in_dim = InDim,out_dim = OutDim)
			handler = get_handler(DATA_NAME)
			strategy = ATLSampling(X_tr, Y_tr, idxs_tr_lb, net, handler, args,test_index=idxs_te_lb,loss = None)

			
			# print info
			print(DATA_NAME)
			#print('SEED {}'.format(SEED))
			print(type(strategy).__name__)
			strategy.data_name = DATA_NAME
			# round 0 accuracy
			strategy.train() #train the model using the training index: idxs_tr_lb
			P = strategy.predict(X_te, Y_te) #test on the active testing set.
			P_hold=strategy.predict(X_hold_te, Y_hold_te)
			strategy.R_noFB = []
			acc = np.zeros(NUM_ROUND+1)
			acc_hold= np.zeros(NUM_ROUND+1)
			acc_fb= np.zeros(NUM_ROUND+1)
			acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
			acc_hold[0]= 1.0 * (Y_hold_te==P_hold).sum().item() / len(Y_hold_te)
			acc_fb[0]= 1.0 * (Y_hold_te==P_hold).sum().item() / len(Y_hold_te)

			print('Round 0\ntesting accuracy {}'.format(acc[0]))
			print('Round 0\nhold testing accuracy {}'.format(acc_hold[0]))
			#first train the test strategy

			strategy.test_init()

			pred_hold_list = []
			for rd in range(1, NUM_ROUND+1):
				print('Round {}'.format(rd))
				
				# query training
				q_idxs = strategy.query_bvs(NUM_TR_QUERY)

				idxs_tr_lb[q_idxs] = True#update the training index.
				strategy.idxs_tr_lb=idxs_tr_lb
				strategy.idxs_te_lb=idxs_te_lb
				
				#retrain the train strategy
				strategy.train()
				# round accuracy
                # compute hold-out test risk
				P = strategy.predict(X_te, Y_te)
				P_hold=strategy.predict(X_hold_te, Y_hold_te)
				acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
				acc_hold[rd] = 1.0 * (Y_hold_te==P_hold).sum().item() / len(Y_hold_te)
				print('testing accuracy {}'.format(acc[rd]))
				print('holdout testing accuracy {}'.format(acc_hold[rd]))
				pred_hold = strategy.predict_prob(X_hold_te,Y_hold_te)
				pred_hold_list.append(pred_hold.detach().cpu().numpy())

				eps = 1e-5
				pred_hold = torch.clamp(pred_hold,eps,1-eps)
				R_hold = torch.zeros(Y_hold_te.shape)
				for c in range(strategy.NClass):
					calc1 = torch.zeros(Y_hold_te.shape)
					calc1[Y_hold_te==c] = 1
					ce1 = torch.mul(calc1,torch.log(pred_hold[:,c]))
					ce2 = torch.mul(1-calc1,torch.log(1-pred_hold[:,c]))
					R_hold = R_hold-torch.mul(calc1,torch.log(pred_hold[:,c]))

				print('hold R:')	
				print(R_hold.mean())
				strategy.R_noFB.append(R_hold.mean())
				strategy.test_init1()

				rhold_list.append(R_hold.mean())
				R_th = strategy.R_th()


				for rd_test in range(1):
					R_nq = strategy.test_query_batch_dif(NUM_ROUND_TEST)

					print('test  '+str(rd_test))
					print(R_nq)
					print('R'+str(R_th))
					print('Rwtest  '+str(rd_test))
					R_test = strategy.R_test()
					R_nq1 = strategy.R_vt_batch(NUM_ROUND_TEST)
					print('Rvt',R_nq1)
					R_vt_list.append(R_nq1.cpu().numpy())

				ind_all = np.array((range(len(strategy.X))))
				ind_test = ind_all[strategy.idxs_te_lb]
				print(strategy.idxs_tr_lb[ind_test])
				print(type(strategy).__name__)
				print(acc)
				print('hold')
				print(acc_hold)
				accs.append(acc)
				hold_accs.append(acc_hold)
				convergeIters.append(convergeIter)
                # update true training and testing risks
				R_tr = strategy.R_tr()
				rtr_list.append(R_tr)
				R_tr = strategy.R_true()
				R_train = strategy.R_train()
				R_test,rtest = strategy.R_test()
				rtrue_list.append(R_tr)
				rtest_list.append(rtest.cpu().detach().numpy())
				rnq_list.append(np.array(strategy.Rtest))
				if NUM_ROUND_TEST>0:
					rnq_finalList.append(np.array(strategy.Rtest)[-1])
				
				#feedback
				strategy.feedBack(int(NUM_ROUND_TEST))
                # compute integrated risk after feedback
				R_nq1_fb = strategy.R_vt_batch(int(NUM_ROUND_TEST/2))
				R_vtfb_list.append(R_nq1_fb.cpu().numpy())

				pred_hold = strategy.predict_prob(X_hold_te,Y_hold_te)
				pred_hold_list.append(pred_hold.detach().cpu().numpy())



				eps = 1e-5
				pred_hold = torch.clamp(pred_hold,eps,1-eps)
				R_hold = torch.zeros(Y_hold_te.shape)
				for c in range(strategy.NClass):
					calc1 = torch.zeros(Y_hold_te.shape)
					calc1[Y_hold_te==c] = 1
					ce1 = torch.mul(calc1,torch.log(pred_hold[:,c]))
					ce2 = torch.mul(1-calc1,torch.log(1-pred_hold[:,c]))
					R_hold = R_hold-torch.mul(calc1,torch.log(pred_hold[:,c]))
				feedback_list.append(R_hold.mean().cpu().numpy())
				R_hold_f = strategy.test_holdout(X_hold_te,Y_hold_te)
				R_fold_list.append(R_hold_f)
				P_hold=strategy.predict(X_hold_te, Y_hold_te)
				acc_fb[rd] = 1.0 * (Y_hold_te==P_hold).sum().item() / len(Y_hold_te)
				print('feedback',acc_fb)
			rtest_var_list.append(rtest_list)
			Acc_var_list.append(acc_hold)
			Acc_FB_var_list.append(acc_fb)
			results_List = [rnq_list,rnq_finalList,rtr_list,rtrue_list,rhold_list,feedback_list,R_fold_list,R_cur_list]
			filep = open('./activetest/'+FILE_NAME+'result1.npy','wb')
			pickle.dump(results_List,filep)
			filep.close()
			mean=list(np.mean(accs,axis=0))
			var=list(np.var(accs,axis=0))
			R_vt_var_list.append(list(np.array(R_vt_list)))
			R_vtfb_var_list.append(list(np.array(R_vtfb_list)))
			R_final_var_list.append(list(np.array(rnq_finalList)))
			R_fb_var_list.append(list(np.array(feedback_list)))
			R_cur_var_list.append(list(np.array(R_cur_list)))
			pred_hold_list_all.append(pred_hold_list)
			q_list_all.append(np.array(strategy.test_qall).reshape(len(strategy.test_qall)))
    # save multiple runs results
	res_var = [R_vt_var_list,R_final_var_list,R_fb_var_list,R_cur_var_list,Acc_var_list,Acc_FB_var_list,R_vtfb_var_list,pred_hold_list_all,rtest_var_list,q_list_all]
	filep = open('./activetest/'+FILE_NAME+'multiple_results.npy','wb')
	pickle.dump(res_var,filep)
	filep.close()
   