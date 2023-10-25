import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize, rosen, rosen_der,Bounds

def initialize_weights(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
		if m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0)
	elif isinstance(m, torch.nn.BatchNorm2d):
		torch.nn.init.constant_(m.weight.data, 1)
		torch.nn.init.constant_(m.bias.data, 0)
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.uniform_(m.weight.data,a=-3,b=3)
		torch.nn.init.constant_(m.bias.data, 0)

class Strategy:
	def __init__(self, X, Y, idxs_lb, net, handler, args,test_index=None,loss='svm',NClass=10,C=1,beta = 1):
		self.X = X
		self.Y = Y
		self.idxs_tr_lb = idxs_lb
		self.net = net
		self.handler = handler
		self.args = args
		self.n_pool = len(Y)
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.idxs_te_lb=test_index
		self.lossType=loss
		self.NClass=NClass
		self.C=C
		self.wvalues = []
		self.N = self.idxs_tr_lb.shape[0]
		self.acq_weights = []
		
	def query(self, n):
		pass

	def update(self, idxs_lb):
		self.idxs_tr_lb = idxs_lb
		self.Phi=self._predict( self.X[self.idxs_tr_lb], self.Y[self.idxs_tr_lb])
		self.Gram= rbf_kernel(self.Phi.numpy())

	def _train(self, epoch, loader_tr, optimizer):
		self.clf.train()
		for batch_idx, (x, y, idxs) in enumerate(loader_tr):
			x, y = x.to(self.device), y.to(self.device)
			optimizer.zero_grad()
			if (self.lossType=='svm'):
				'''
				out, e1, w = self.clf(x) #out puts are h, a ,w
				l2_lambda = 0.05
				l2_reg = torch.tensor(0.).cuda()
				for param in self.clf.parameters():
					l2_reg += torch.norm(param)
				resTensor=torch.zeros(self.NClass,e1[0].shape[0])
				for k in range(self.NClass):
					resTensor[k,:]=e1[k][:,0]                    
				m = torch.nn.Softmax(dim=0)    
				a=m(resTensor)
				a=a.T
				self.temPre=a
				self.temY=y
				loss = F.cross_entropy(a.cuda(), y.cuda())    
				loss.backward()   
				optimizer.step()
				#print(loss)
				
				'''
				totalLoss=torch.tensor(0.).cuda()
				for k in range(self.NClass):#iteratively update k-losses
					#print('+++++++++++++++++++++++k=%d'%(k))
					out, e1, w = self.clf(x) #out puts are h, a ,w
					t=torch.zeros(y.shape)
					t=t-1
					t[torch.where(y==k)]=1#convert y to {1,-1} for each k.              
					t=t.to(self.device)
					zeros=torch.zeros(y.shape).to(self.device)
					ones=torch.ones(y.shape).to(self.device)
					#self.w=w[k]
					#part1=0.5*(torch.inner(w[k],w[k]))
					#part2=self.C*torch.sum(torch.square(torch.max(zeros,ones-e1[k]*t)))
					self.temY=y
					l2_lambda = 0.0005
					l2_reg = torch.tensor(0.).cuda()
					
					for param in self.clf.parameters():
						l2_reg += torch.norm(param)
					if k==1:
						hinge = self.C*torch.sum(torch.square(torch.max(zeros,ones-e1[k].view(y.shape)*t)))
						print('hinge')
						print(hinge)
						print(l2_lambda * l2_reg )
					loss_k=0.5*(torch.inner(w[k][:,:-1],w[k][:,:-1])) + self.C*torch.sum(torch.square(torch.max(zeros,ones-e1[k].view(y.shape)*t))) +l2_lambda * l2_reg                    
					totalLoss+=loss_k[0,0]
					# totalLoss+=l2_reg+l2_lambda * l2_reg 
					#loss_k=part1+part2
					#print('loss part1 %f'%(part1))
					#print('loss part2 %f'%(part2))
					#self.temE=e1
					#self.temW=w[k]
					#self.temOut=out
					#optimizer.zero_grad()
					#loss_k.backward(retain_graph=True)
					#optimizer.step()
				# self.wvalues.append(w)
				optimizer.zero_grad()
				totalLoss.backward()
				#print(totalLoss)
				optimizer.step()
					#print(k)
					
					
			else:    
				out, e1 = self.clf(x)
				loss = F.cross_entropy(out, y)
				loss.backward()
				self.loss=loss.cpu().detach().numpy()
				optimizer.step()
			#print(loss)
	
	def _train1(self, epoch, loader_tr, optimizer):
		self.clf.train()
		for batch_idx, (x, y, idxs) in enumerate(loader_tr):
			# self.X = x
			# self.Y = y
			y=y.type(torch.FloatTensor)
			x, y = x.to(self.device), y.to(self.device)
			optimizer.zero_grad()
			if (self.lossType=='svm'):
				'''
				out, e1, w = self.clf(x) #out puts are h, a ,w
				l2_lambda = 0.05
				l2_reg = torch.tensor(0.).cuda()
				for param in self.clf.parameters():
					l2_reg += torch.norm(param)
				resTensor=torch.zeros(self.NClass,e1[0].shape[0])
				for k in range(self.NClass):
					resTensor[k,:]=e1[k][:,0]                    
				m = torch.nn.Softmax(dim=0)    
				a=m(resTensor)
				a=a.T
				self.temPre=a
				self.temY=y
				loss = F.cross_entropy(a.cuda(), y.cuda())    
				loss.backward()   
				optimizer.step()
				#print(loss)
				
				'''
				totalLoss=torch.tensor(0.).cuda()
				
				out, e1, w = self.clf(x)
				self.EE = e1
				self.YY = y
				self.wvalues.append(w)
				for k in range(self.NClass):
					t=torch.zeros(y.shape)
					# t=t-1
					t[torch.where(y==k)]=1#convert y to {1,-1} for each k.              
					t=t.to(self.device)
					m = torch.nn.Sigmoid()
					e11 = m(e1[k])
					# loss = torch.nn.BCELoss()
					loss = torch.nn.MSELoss()
					totalLoss+= loss(e11.view(y.shape).cuda(), y.cuda())
				
				# e11 = F.sigmoid(e1[0])

				# break
				# totalLoss=F.cross_entropy(e1[0].cuda(), y.cuda())#out puts are h, a ,w
				# totalLoss = F.binary_cross_entropy(e11.view(y.shape).cuda(), y.cuda())
				print(totalLoss)
				# loss=torch.nn.BCEWithLogitsLoss()
				# totalLoss = loss(e1[0].view(y.shape).cuda(), y.type(torch.FloatTensor).cuda())
				#out puts are h, a ,w

				# l2_lambda = 0.05
				# l2_reg = torch.tensor(0.).cuda()
				# for param in self.clf.parameters():
				#     l2_reg += torch.norm(param)
				# totalLoss+=l2_reg

				# loss_k=0.5*(torch.inner(w,w)) + self.C*torch.sum(torch.square(torch.max(zeros,ones-e1.view(y.shape)*t)))                    
				# totalLoss+=loss_k[0,0]

				optimizer.zero_grad()
				totalLoss.backward()
				#print(totalLoss)
				optimizer.step()
					#print(k)
					
					
			else:    
				out, e1 = self.clf(x)
				loss = F.cross_entropy(out, y)
				loss.backward()
				self.loss=loss.cpu().detach().numpy()
				optimizer.step()
			#print(loss)
	# def train1(self,trainInd=None):
	# 	 n_epoch = self.args['n_epoch']
	# 	 if(self.lossType=='svm'):
	# 		 self.clf = self.net(numClass=self.NClass).to(self.device)
	# 	 else:
	# 		 self.clf = self.net().to(self.device)
	# 	 # self.clf.apply(initialize_weights)    
	# 	 optimizer = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
	# 	 if(trainInd is None):
	# 		 idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb]
	# 	 elif(trainInd == 'both'):
	# 		 idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb+self.idxs_te_lb]
	# 	 loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
	# 						 shuffle=True, **self.args['loader_tr_args'])
	# 	 for epoch in range(1, n_epoch+1):
	# 		 self._train1(epoch, loader_tr, optimizer)           

	def train(self,trainInd=None):
		n_epoch = self.args['n_epoch']
		if(self.lossType=='svm'):
			self.clf = self.net(numClass=self.NClass).to(self.device)
		else:
			self.clf = self.net().to(self.device)
		# self.clf.apply(initialize_weights)
		optimizer = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
		if(trainInd is None):
			idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb]
		elif(trainInd == 'both'):
			idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb+self.idxs_te_lb]
		loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
							shuffle=True, **self.args['loader_tr_args'])
		for epoch in range(1, n_epoch+1):
			self._train(epoch, loader_tr, optimizer)
		

	
	def _predict(self, X, Y):#used to get the train gram(Phi)
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])
		self.clf.eval()
		Phi = torch.zeros(len(Y),50)#e(hidden layer has dim 50)
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				if (self.lossType=='svm'):
					out, e1,w = self.clf(x)
				else:
					out, e1 = self.clf(x)
				Phi[idxs]=e1.cpu()
		return Phi

	def predict(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])
		self.clf.eval()
		P = torch.zeros(len(Y), dtype=Y.dtype)
		Phi = torch.zeros(len(Y),self.clf.get_embedding_dim())#e(hidden layer has dim 50)
		#outs=torch.zeros(len(Y),10, dtype=Y.dtype)
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				if (self.lossType=='svm'):
					out, e1,w = self.clf(x)
					pred=np.argmax(np.array([i.cpu().numpy() for i in e1])[:,:,0],axis=0)
					#outs[idxs]=out.cpu()
					P[idxs] = torch.tensor(pred)
				else:
					out, e1 = self.clf(x)
					Phi[idxs]=e1.cpu()
					pred = out.max(1)[1]
					P[idxs] = pred.cpu()
					self.Phi=Phi
		return P
	def predict0(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])
		self.clf.eval()
		P = torch.zeros(len(Y), dtype=Y.dtype)
		Phi = torch.zeros(len(Y),50)#e(hidden layer has dim 50)
		#outs=torch.zeros(len(Y),10, dtype=Y.dtype)
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				if (self.lossType=='svm'):
					out, e1,w = self.clf(x)
					pred=np.argmax(np.array([i.cpu().numpy() for i in e1])[:,:,0],axis=0)
					#outs[idxs]=out.cpu()
					P[idxs] = torch.tensor(pred)
				else:
					out, e1 = self.clf(x)
					Phi[idxs]=e1.cpu()
					pred = out.max(1)[1]
					P[idxs] = pred.cpu()
					self.Phi=Phi
		return out,e1,w
	def predict1(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])
		self.clf.eval()
		P = torch.zeros(len(Y), dtype=Y.dtype)
		Phi = torch.zeros(len(Y),50)#e(hidden layer has dim 50)
		#outs=torch.zeros(len(Y),10, dtype=Y.dtype)
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				if (self.lossType=='svm'):
					out, e1,w = self.clf(x)
				   
		return out,e1,w
	def predict_prob(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.eval()
		probs = torch.zeros([len(Y), self.NClass])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf(x)
				prob = F.softmax(out, dim=1)
				probs[idxs] = prob.cpu()
		
		return probs

	def predict_prob_dropout(self, X, Y, n_drop):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.train()
		probs = torch.zeros([len(Y), len(np.unique(Y))])
		for i in range(n_drop):
			print('n_drop {}/{}'.format(i+1, n_drop))
			with torch.no_grad():
				for x, y, idxs in loader_te:
					x, y = x.to(self.device), y.to(self.device)
					out, e1 = self.clf(x)
					prob = F.softmax(out, dim=1)
					probs[idxs] += prob.cpu()
		probs /= n_drop
		
		return probs

	def predict_prob_dropout_split(self, X, Y, n_drop):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.train()
		probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
		for i in range(n_drop):
			print('n_drop {}/{}'.format(i+1, n_drop))
			with torch.no_grad():
				for x, y, idxs in loader_te:
					x, y = x.to(self.device), y.to(self.device)
					out, e1 = self.clf(x)
					probs[i][idxs] += F.softmax(out, dim=1).cpu()
		
		return probs

	def get_embedding(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.eval()
		embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf(x)
				embedding[idxs] = e1.cpu()
		
		return embedding

	def _trainSVM(self, epoch, loader_tr, optimizer):
		self.clf.train()
		for batch_idx, (x, y, idxs) in enumerate(loader_tr):
			# self.X = x
			# self.Y = y
			# y=y.type(torch.FloatTensor)
			x, y = x.to(self.device), y.to(self.device)
			# x, y = x.to(self.device), y.to(self.device)
			totalLoss=torch.tensor(0.).to(self.device)
			for k in range(self.NClass):
				out, e1, w = self.clf(x)
				t=torch.zeros(y.shape)
				t=t-1
				t[torch.where(y==k)]=1#convert y to {1,-1} for each k.              
				t=t.to(self.device)
				zeros=torch.zeros(y.shape).to(self.device)
				ones=torch.ones(y.shape).to(self.device)
				part2= self.C*torch.sum(torch.square(torch.max(zeros,ones-e1[k].view(y.shape)*t)))
				part1=0.5*(torch.inner(w[k],w[k]))[0][0]
				
				#print(part2)
				#print(part1)
				loss_k=0.5*(torch.inner(w[k],w[k])[0][0]) + self.C*torch.sum(torch.square(torch.max(zeros,ones-e1[k].view(y.shape)*t))) #+l2_lambda * l2_reg                    
				#print('k th loss %f'%(loss_k))
				loss_k.backward(retain_graph=True)
				optimizer.step()
				self.htr = out
			# print(totalLoss)



				
	# def trainSVM(self,trainInd=None):
	# 	 n_epoch = self.args['n_epoch']
	# 	 if(self.lossType=='svm'):
	# 		 self.clf = self.net(NClass=self.NClass).to(self.device)
	# 	 else:
	# 		 self.clf = self.net().to(self.device)
	# 	 # self.clf.apply(initialize_weights)    
	# 	 optimizer = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
	# 	 if(trainInd is None):
	# 		 idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb]
	# 	 elif(trainInd == 'both'):
	# 		 idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb+self.idxs_te_lb]
	# 	 loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
	# 						 shuffle=True, **self.args['loader_tr_args'])
	# 	 for epoch in range(1, n_epoch+1):
	# 		 self._trainSVM(epoch, loader_tr, optimizer) 




	# def predictSVM(self, X, Y):
	# 	loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
	# 						shuffle=False, **self.args['loader_te_args'])
	# 	self.clf.eval()
	# 	P = torch.zeros(len(Y), dtype=Y.dtype)
	# 	Phi = torch.zeros(len(Y),50)#e(hidden layer has dim 50)
	# 	#outs=torch.zeros(len(Y),10, dtype=Y.dtype)
	# 	with torch.no_grad():
	# 		for x, y, idxs in loader_te:
	# 			x, y = x.to(self.device), y.to(self.device)
	# 			if (self.lossType=='svm'):
	# 				out, e1,w = self.clf(x)
	# 				# pred=np.argmax(np.array([i.cpu().numpy() for i in e1])[:,:,0],axis=0)
	# 				#outs[idxs]=out.cpu()
	# 				pred=np.argmax(np.array([i.cpu().numpy() for i in e1])[:,:,0],axis=0)
	# 				P[idxs] = torch.tensor(pred)
	# 				# print( 1.0 * (Y_te==P).sum().item() / len(Y_te))
	# 			else:
	# 				out, e1 = self.clf(x)
	# 				Phi[idxs]=e1.cpu()
	# 				pred = out.max(1)[1]
	# 				P[idxs] = pred.cpu()
	# 				self.Phi=Phi
	# 	return P,out,e1,w

	def get_dualcoef(self,X,Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False,**self.args['loader_te_args'])
		self.clf.eval()
		
		with torch.no_grad():
			for x,y,idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1, w = self.clf(x)
				wList = []
				n = out.shape[0]
				m = out.shape[1]
				h = out.cpu().detach().numpy()
				for i in range(self.NClass):
					wList.append(w[i].cpu().detach().numpy())
				alpha = []
				print(1)
				for i in range(self.NClass):
					t = np.zeros((n,1))
					t = t-1
					t[torch.where(y.cpu()==i)]=1
					tt = np.repeat(t,m,1)
					ww = wList[i]
					Xt = h*tt
					print(n)
					print(m)
					funres = lambda x: (np.matmul(np.matmul(x,Xt)-ww,(np.matmul(x,Xt)-ww).T)[0,0])
					cons = ( {'type':'eq','fun': lambda x: np.matmul(t.T,x)})
					bnds = Bounds(0,self.C)

					scires = minimize(funres, np.ones((n,1)), method='SLSQP', bounds = bnds,
							constraints=cons)
					xsol2 = scires.x
					lossres = scires.fun
					print(i)
					alpha.append(xsol2)
		return alpha
	
	def get_alpha(self):
		ind_tr = np.arange(self.n_pool)[self.idxs_tr_lb]
		Xget = self.X[ind_tr]
		Yget = self.Y[ind_tr]
		alpha = self.get_dualcoef(Xget,Yget)
		return alpha
	
	def predict_probSVM(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.eval()
		probs = torch.zeros([len(Y), len(np.unique(Y))])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1, w = self.clf(x)
				logit = torch.cat((e1[0],e1[1]),dim = 1)
				for i in range(2,self.NClass):
					logit = torch.cat((logit,e1[i]),dim=1)
				prob = F.softmax(logit, dim=1)
				probs[idxs] = prob.cpu()
		
		return probs
   
	
	def get_dm(self, Xtr, Ytr, Xun, Yun):
		alpha = self.get_alpha()
		
		loader_te = DataLoader(self.handler(Xun, Yun, transform=self.args['transform']),
							shuffle=False,**self.args['loader_te_args'])
		self.clf.eval()
		embedding = torch.zeros([len(Yun), self.clf.get_embedding_dim()])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1, w = self.clf(x)
				embedding[idxs] = out.cpu()

		loader_te = DataLoader(self.handler(Xtr, Ytr, transform=self.args['transform']),
							shuffle=False,**self.args['loader_te_args'])
		self.clf.eval()
		htr = torch.zeros([len(Ytr), self.clf.get_embedding_dim()])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1, w = self.clf(x)
				htr[idxs] = out.cpu()


		kernel = np.matmul(embedding.cpu().numpy(), htr.cpu().numpy().T)
		dm_List = []
		for i in range(self.NClass):
			alpha_pos = copy.deepcopy(alpha[i])
			alpha_pos[np.where(alpha[i]<0)] = 0
			dm = np.matmul(kernel,alpha_pos)
			dm_List.append(dm)
		return dm_List
	
	# def get_dm(self,Xun,Yun):
	#     alpha = self.get_alpha()
		
	#     loader_te = DataLoader(self.handler(Xun, Yun, transform=self.args['transform']),
	#                         shuffle=False,**self.args['loader_te_args'])
	#     self.clf.eval()
	#     embedding = torch.zeros([len(Yun), self.clf.get_embedding_dim()])
	#     with torch.no_grad():
	#         for x, y, idxs in loader_te:
	#             x, y = x.to(self.device), y.to(self.device)
	#             out, e1, w = self.clf(x)
	#             embedding[idxs] = out.cpu()

	#     kernel = np.matmul(embedding.cpu().numpy(), self.htr.cpu().detach().numpy().T)
	#     self.kernel = kernel
	#     dm_List = []
	#     for i in range(self.NClass):
	#         alpha_pos = copy.deepcopy(alpha[i])
	#         alpha_pos[np.where(alpha[i]>0)] = 0
	#         self.alpha_pos = alpha_pos
	#         dm = np.matmul(kernel,alpha_pos)
	#         dm_List.append(dm)
	#     return dm_List
	def _train_bnn(self, epoch, loader_tr, optimizer, criterion):
		train_loss = 0.0
		wc = 1 / (len(loader_tr.dataset) // self.args['loader_tr_args']['batch_size'])
		self.clf.train()
		self.criterion = criterion
		for batch_idx, (data, labels, idxs) in enumerate(loader_tr):
			data, labels = data.to(self.device), labels.to(self.device)

			optimizer.zero_grad()

			outputs, e1 = self.clf(data)
			loss = self.clf.elbo(data, labels, self.criterion, n_samples=5, w_complexity=wc)
			train_loss += loss.item() * data.size(0)

			loss.backward()
			optimizer.step()
		train_loss /= len(loader_tr.dataset)
		return train_loss


	def train_bnn(self, trainInd=None):
		n_epoch = self.args['n_epoch']
		self.clf = self.net.to(self.device)
		optimizer = optim.Adam(self.clf.parameters(),**self.args['optimizer_adam_args'])
		if(trainInd is None):
			idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb]
		elif(trainInd == 'both'):
			idxs_train = np.arange(self.n_pool)[self.idxs_tr_lb+self.idxs_te_lb]
		loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
							shuffle=True, **self.args['loader_tr_args'])
		criterion = torch.nn.CrossEntropyLoss(reduction='sum')
		for epoch in range(1, n_epoch+1):
			self._train_bnn(epoch, loader_tr, optimizer,criterion)


	def test_bnn(model, testloader, device):

		correct = 0
		test_loss = 0.0

		model.eval()
		for batch_idx, (data, labels) in enumerate(testloader):
			data, labels = data.to(device), labels.to(device)

			outputs = model(data)
			loss = F.cross_entropy(outputs, labels)
			test_loss += loss.item() * data.size(0)

			preds = outputs.argmax(dim=1, keepdim=True)
			correct += preds.eq(labels.view_as(preds)).sum().item()

		test_loss /= len(testloader.dataset)
		accuracy = correct / len(testloader.dataset)

		return test_loss, accuracy

	def test_holdout(self,X_hold,Y_hold):
		n_hold = len(Y_hold)
		n_fold = int(n_hold/5)
		hold_ind = np.arange(n_hold)
		np.random.shuffle(hold_ind)
		R_hold = []
		for i in range(5):
			X_fold = X_hold[list(np.array(hold_ind)[i*n_fold:(i+1)*n_hold])]
			Y_fold = Y_hold[list(np.array(hold_ind)[i*n_fold:(i+1)*n_hold])]
			pred = self.predict_prob(X_fold,Y_fold)
			eps = 1e-5
			pred = torch.clamp(pred,eps,1-eps)
			R_fold = torch.zeros(Y_fold.shape)
			for c in range(self.NClass):
				calc1 = torch.zeros(Y_fold.shape)
				calc1[Y_fold==c] = 1
				R_fold = R_fold-torch.mul(calc1,torch.log(pred[:,c]))
			R_fold = R_fold.mean()
			R_hold.append(R_fold)
		print('hold R:')
		print(np.mean(R_hold))
		return np.mean(R_hold),R_hold

	def featureout(self, X, Y):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		self.clf.eval()
		if self.data_name=='MNIST':
			self.edim = 50
		else:
			self.edim=50
		featureout = torch.zeros([len(Y), self.NClass])
		featuree= torch.zeros([len(Y), self.edim])
		with torch.no_grad():
			for x, y, idxs in loader_te:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf(x)
				prob = F.softmax(out, dim=1)
				featureout[idxs] = out.cpu()
				featuree[idxs] = e1.cpu()
		
		return featureout,featuree
