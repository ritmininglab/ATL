from ast import AsyncFunctionDef
import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from datetime import datetime
import pickle
import copy

class ATLSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args,test_index=None,loss=None,NClass=10,C=1,beta=1,tor = 1e-4):
		super(ATLBSampling, self).__init__(X, Y, idxs_lb, net, handler, args,test_index=test_index,loss=loss,NClass=NClass,C=C,beta=beta)
		# tolerance
		self.tor = tor

    #estimate pool risk (cross-entropy loss)
	def R_th(self):	
        total = self.X
		pred = self.predict_prob(total,self.Y)
        
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,self.Y)
		r = torch.zeros(self.Y.shape)
		for c in range(self.NClass):
			calc1 = -torch.log(pred[:,c])
			r = r+torch.mul(calc1,pred[:,c])
		R = r.mean()
		return R

    #estimate pool difference
	def Dif_th(self):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]
		total = self.X[idxs_unlabeled]
		total_Y = self.Y[idxs_unlabeled]		
		R_th = self.R_nq
		pred = self.predict_prob(total,total_Y)
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,total_Y)
		print(pred.shape)
		p,q = self.q(R_th)
		print(q.shape)
		r = torch.zeros(total_Y.shape)
		for c in range(self.NClass):
			calc1 = torch.square(-torch.log(pred[:,c])-R_th)
			r = r+torch.mul(calc1,pred[:,c])
		R = torch.mul(1/q,r)
		R = R/((1/q).sum())
		R = R.sum()
		R = torch.sqrt(R)
		return R
    
	# True loss based on true labels
	def R_true(self):		
		total = self.X
		total_Y = self.Y
		pred = self.predict_prob(total,total_Y)
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,total_Y)
		r = torch.zeros(self.Y.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(total_Y.shape)
			calc1[total_Y==c] = 1
			logprob = -torch.log(pred[:,c])
			calc1 = torch.mul(calc1,logprob)
			r = r+calc1
		R = r.mean()
		return R

	# true training loss
	def R_tr(self):		
		total = self.X
		pred = self.predict_prob(total,self.Y)
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,self.Y)
		r = torch.zeros(self.Y.shape)
		for c in range(self.NClass):
			calc1 = -torch.log(pred[:,c])
			r = r+torch.mul(calc1,pred[:,c])
		R = r.mean()
		return R

	# ARE testing proposal q
	def q(self,R_th):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]
		total = self.X[idxs_unlabeled]
		total_Y = self.Y[idxs_unlabeled]
		eps = 1e-5
		pred = self.predict_prob(total,total_Y)
		pred = torch.clamp(pred,eps,1-eps)
		log_probs = torch.log(pred)
		U = torch.ones(U.shape)

		pred_l = self.predict(total,total_Y)
		q = torch.zeros(total_Y.shape)
		# # compute (l-R_th) over all classes
		for c in range(self.NClass):
			calc1 = torch.log(pred[:,c])
			q = q+torch.mul(torch.square((calc1-R_th)),pred[:,c])
		q = torch.sqrt(q)
		eps = 1e-5
		q = torch.mul(U,q)
		q = F.normalize(q,p=1,dim=0)
		q = torch.clamp(q,eps,1-eps)
		q = F.normalize(q,p=1,dim=0)
		return U,q


    #BVS sampling for AL
	def query_bvs(self, n):
		if(self.idxs_te_lb is None):
			idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]    
		else:
			idxs_unlabeled = np.arange(self.n_pool)[~(self.idxs_tr_lb+self.idxs_te_lb)]        
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		probs_sorted, idxs = probs.sort(descending=True)
		U = probs_sorted[:, 0] - probs_sorted[:,1]
		return idxs_unlabeled[U.sort()[1][:n]]

    # Active testing initialization
	def test_init(self):
		self.test_qall = []
		self.Rtest = []
		self.test_x = None
		self.test_y = None
		self.test_pall = []
		self.test_ind = []
		self.R_estList = []
		self.R_es = self.R_th()
		self.R_nq = 0
		self.LossTest = []
		self.ps='1'
		self.R_cur = []

    # Quiz initialization (reset R_th)
	def test_init1(self):
		self.R_es = (self.R_train()*np.sum(self.idxs_tr_lb)+self.R_es*self.n_pool+\
		 	self.R_nq*len(self.test_qall))/(self.n_pool)

	

	# # batch active quiz selection
	def test_query_batch(self,n):
		# # estimated R with R_tr and R_nq
		R = self.R_es
		print('computing R')
		print(R)
		self.R_estList.append(1/R)
		p,q = self.q(R)
		q = F.normalize(q,p=1,dim=0)
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]
		total = self.X[idxs_unlabeled]
		total_Y = self.Y[idxs_unlabeled]

        # numpy normalize to avoid errors
		norm = np.linalg.norm(q.cpu().numpy(),ord=1)
		q = q/norm

		poolInd = idxs_unlabeled
		test_q = []

        # test sampling: n samples
		for i in range(n):
			sample=np.random.choice(poolInd,p=q.cpu().numpy())
			sample_ind = np.where(poolInd==sample)

			self.idxs_te_lb[sample] = True
			test_q.append(q[sample_ind])
			self.test_qall.append(q[sample_ind].cpu().numpy())
			self.test_pall.append(p[sample_ind].cpu().numpy())
			self.test_ind.append(sample)
			self.test_x = self.X[self.test_ind]
			self.test_y = self.Y[self.test_ind]

		pred_sample = self.predict_prob(self.test_x,self.test_y)
		eps = 1e-5
		pred_sample = torch.clamp(pred_sample,eps,1-eps)
		log_probs = torch.log(pred_sample)
        # uniform pool distribution
		U = torch.ones(U.shape)

		
		Y_sample = self.test_y
		R_sample = torch.zeros(Y_sample.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(Y_sample.shape)
			calc1[Y_sample==c] = 1
			ce1 = torch.mul(calc1,torch.log(pred_sample[:,c]))
			ce2 = torch.mul(1-calc1,torch.log(1-pred_sample[:,c]))
			R_sample = R_sample-torch.mul(calc1,torch.log(pred_sample[:,c]))

		R_sample[torch.isnan(R_sample)]=0
		self.R_sample = R_sample
		self.LossTest.append(R_sample)

		# compute current R
		R_nq = (torch.mul(R_sample,U)/torch.tensor(self.test_qall)).sum()
		R_nq = R_nq/((U/torch.tensor(self.test_qall)).sum())
		self.Rtest.append(R_nq.cpu().numpy())
		self.R_nq = R_nq
        # compute weights V
		V = self.Dif_th_new()
		self.R_estList.append(1/V)
		return R_nq

    # random test query
	def test_query_random_batch(self,n):
		# # estimated R with R_tr and R_nq
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]
		total = self.X[idxs_unlabeled]
		total_Y = self.Y[idxs_unlabeled]
		poolInd = np.arange(len(idxs_unlabeled))
		np.random.shuffle(poolInd)
		sample_ind = poolInd[:n]
		sample = idxs_unlabeled[sample_ind]
		self.idxs_te_lb[sample] = True
		q=np.ones(n)
		self.test_qall+=list(q)
		self.test_pall+=list(q)
		self.test_ind+=list(sample)

		self.test_x = self.X[self.test_ind]
		self.test_y = self.Y[self.test_ind]

		pred_sample = self.predict_prob(self.test_x,self.test_y)
		eps = 1e-5
		pred_sample = torch.clamp(pred_sample,eps,1-eps)
		log_probs = torch.log(pred_sample)

		U = torch.ones(U.shape)

		
		Y_sample = self.test_y
		R_sample = torch.zeros(Y_sample.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(Y_sample.shape)
			calc1[Y_sample==c] = 1
			ce1 = torch.mul(calc1,torch.log(pred_sample[:,c]))
			ce2 = torch.mul(1-calc1,torch.log(1-pred_sample[:,c]))
			R_sample = R_sample-torch.mul(calc1,torch.log(pred_sample[:,c]))
		R_sample[torch.isnan(R_sample)]=0
		self.R_sample = R_sample
		self.LossTest.append(R_sample)		
		R_nq = (torch.mul(R_sample,U)/torch.tensor(self.test_qall)).sum()
		R_nq = R_nq/((U/torch.tensor(self.test_qall)).sum())
		self.Rtest.append(R_nq.cpu().numpy())
		self.R_nq = R_nq
		return R_nq

    # compute integrated test risk (apply weights V)
	def R_vt_batch(self,n):
		R_estList = self.R_estList

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_tr_lb]
		total = self.X[idxs_unlabeled]
		total_Y = self.Y[idxs_unlabeled]
		q = self.test_qall
		print(q)
		norm = np.linalg.norm(np.array(q),ord=1)
		q = np.array(q)/norm
		p = self.test_pall
		norm = np.linalg.norm(np.array(p),ord=1)
		p = np.array(p)/norm
		testind = self.test_ind

		poolInd = idxs_unlabeled
		test_q = []

		test_x = self.test_x
		test_y = self.test_y

		pred_sample = self.predict_prob(self.test_x,self.test_y)
		eps = 1e-5
		pred_sample = torch.clamp(pred_sample,eps,1-eps)
		log_probs = torch.log(pred_sample)
		U = torch.abs((pred_sample*log_probs)).sum(1)
		U = F.normalize(U,p=1,dim=0)
		if self.ps=='1/U':
			U = 1/U
		elif self.ps=='1_U':
			U = 1-U
		elif self.ps=='1':
			U = torch.ones(U.shape)
		U = F.normalize(U,p=1,dim=0)
		
		Y_sample = self.test_y
		R_sample = torch.zeros(Y_sample.shape)
		V_sample = torch.zeros(Y_sample.shape)
		Rall = np.sum(self.R_estList)
		num = 0
		for R0 in self.R_estList:
			v = R0/Rall
			V_sample[num*n:(num+1)*n] = v
			num+=1

		for c in range(self.NClass):
			calc1 = torch.zeros(Y_sample.shape)
			calc1[Y_sample==c] = 1
			ce1 = torch.mul(calc1,torch.log(pred_sample[:,c]))
			ce2 = torch.mul(1-calc1,torch.log(1-pred_sample[:,c]))
			R_sample = R_sample-torch.mul(calc1,torch.log(pred_sample[:,c]))
		R_sample[torch.isnan(R_sample)]=0

		R_sample1 = torch.mul(R_sample,V_sample)
		R_nq1 = (torch.mul(R_sample1,U)/torch.tensor(self.test_qall)).sum()
		R_nq1 = R_nq1/((U/torch.tensor(self.test_qall)).sum())
		self.R_nq1 = R_nq1
		return R_nq1


	def R_test(self):		
		total = self.X[self.idxs_te_lb]
		total_Y = self.Y[self.idxs_te_lb]
		
		pred = self.predict_prob(total,total_Y)
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,total_Y)
		r = torch.zeros(total_Y.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(total_Y.shape)
			calc1[total_Y==c] = 1
			logprob = -torch.log(pred[:,c])
			calc1 = torch.mul(calc1,logprob)
			r = r+calc1
		R = r.mean()
		return R,r

	def R_train(self):		
		total = self.X[self.idxs_tr_lb]
		total_Y = self.Y[self.idxs_tr_lb]
		
		pred = self.predict_prob(total,total_Y)
		# print(pred.sum(dim=1))
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,total_Y)
		r = torch.zeros(total_Y.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(total_Y.shape)
			calc1[total_Y==c] = 1
			logprob = -torch.log(pred[:,c])
			calc1 = torch.mul(calc1,logprob)
			r = r+calc1
		R = r.mean()
		return R

    # active feedback
	def feedBack(self,n):
		total =  self.test_x
		total_Y = self.test_y
		idxs_test = np.arange(self.n_pool)[self.idxs_te_lb]
		pred = self.predict_prob(total,total_Y)
		eps = 1e-5
		pred = torch.clamp(pred,eps,1-eps)
		pred_l = self.predict(total,total_Y)
		r = torch.zeros(total_Y.shape)
		for c in range(self.NClass):
			calc1 = torch.zeros(total_Y.shape)
			calc1[total_Y==c] = 1
			logprob = -torch.log(pred[:,c])
			calc1 = torch.mul(calc1,logprob)
			r = r+calc1
		r = torch.mul(torch.tensor(self.test_qall).view(total_Y.shape),r)
		u = torch.ones(r.shape)
		r = torch.mul(u,r)
		r = -r
		feedBack = list(np.array(self.test_ind)[r.sort()[1][:int(n/2)]])
		#delete all test samples that are used for feedback
		print('fb',feedBack)
		for fb_ind in feedBack:
			test_cur = np.zeros(len(self.test_ind), dtype=bool)
			test_cur[self.test_ind==fb_ind] = True

			self.test_ind = list(np.array(self.test_ind)[~test_cur])

			self.test_qall = list(np.array(self.test_qall)[~test_cur])
			self.test_pall = list(np.array(self.test_pall)[~test_cur])
		self.idxs_te_lb[feedBack]=False
		self.idxs_tr_lb[feedBack]=True
		self.idxs_te_lb[self.test_ind] = True
		self.test_x = self.X[self.test_ind]
		self.test_y = self.Y[self.test_ind]


	def countLB_train(self):
		return np.sum(self.idxs_tr_lb)

	def countLB_test(self):
		return np.sum(self.idxs_te_lb)

	def countLB_all(self):
		return np.sum((self.idxs_te_lb+self.idxs_tr_lb))