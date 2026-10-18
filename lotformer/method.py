import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);self.token=nn.Linear(4,w)
        self.inducing=nn.Parameter(torch.randn(c.get('inducing',8),w)*.1);self.attention=nn.MultiheadAttention(w,4,batch_first=True)
        self.outer=nn.MultiheadAttention(w,4,batch_first=True);self.reliability=nn.Linear(2*w+1,1);self.score=nn.Linear(w,1);self.head=nn.Linear(w,9)
    def predict_at(self,b,k):
        x=b['x'].float()[:,:k];n,wafer,die,_=x.shape;tokens=self.token(x).reshape(n*wafer,die,-1);valid=x[...,3].reshape(n*wafer,die)>0
        valid=valid.clone();valid[:,0]=True;inducing=self.inducing[None].expand(n*wafer,-1,-1)
        for _ in range(2):
            induced=self.attention(inducing,tokens,tokens,key_padding_mask=~valid)[0];tokens=tokens+self.attention(tokens,induced,induced)[0]
        h=(tokens*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True);h=h.reshape(n,wafer,-1)
        previous=(h.cumsum(1)-h)/torch.arange(1,wafer+1,device=h.device)[None,:,None]
        quality=x[...,3].mean(-1,keepdim=True);q=self.reliability(torch.cat([h,previous,quality],-1)).sigmoid()
        context=h+self.outer(h,h,h)[0];score=self.score(context).squeeze(-1)
        if self.c['variant'] not in {'baseline','no_reliability'}:score=score+q.squeeze(-1).clamp_min(1e-6).log()
        weights=score.softmax(-1)
        if self.c['variant'] not in {'baseline','no_cap'}:
            cap=min(1.,self.c.get('cap',2.5)/wafer);low=score.min(-1,keepdim=True).values-1;high=score.max(-1,keepdim=True).values
            for _ in range(40):
                middle=(low+high)/2;mass=(score-middle).clamp(0,cap).sum(-1,keepdim=True);low=torch.where(mass>1,middle,low);high=torch.where(mass>1,high,middle)
            weights=(score-(low+high)/2).clamp(0,cap);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        pi,mu,raw=self.head((weights[:,:,None]*context).sum(1)).chunk(3,-1);pi=pi.softmax(-1);sigma=F.softplus(raw)+.001
        passed=(x[...,2]*x[...,3]).sum((1,2));observed=x[...,3].sum((1,2));plan=b['planned_dies'].float().clamp_min(1);lower=passed/plan;upper=(passed+plan-observed)/plan
        if self.c['variant']=='baseline':lower=torch.zeros_like(lower);upper=torch.ones_like(upper)
        delta=(upper-lower).clamp_min(0);nodes=torch.linspace(.5/64,1-.5/64,64,device=x.device);normal=torch.special.ndtri(nodes)
        samples=lower[:,None,None]+delta[:,None,None]*torch.sigmoid(mu[:,:,None]+sigma[:,:,None]*normal)
        mean=(samples.mean(-1)*pi).sum(-1);flat=samples.flatten(1);mass=pi[:,:,None].expand_as(samples).flatten(1)/64;order=flat.argsort(-1);sorted_sample=flat.gather(1,order);cdf=mass.gather(1,order).cumsum(-1)
        quantile=lambda t:sorted_sample.gather(1,(cdf<t).sum(-1,keepdim=True).clamp_max(flat.shape[1]-1)).squeeze(-1)
        threshold=self.c.get('low_yield',.7);coordinate=((threshold-lower)/delta.clamp_min(1e-6)).clamp(1e-6,1-1e-6);risk=(pi*torch.special.ndtr((torch.logit(coordinate)[:,None]-mu)/sigma)).sum(-1)
        risk=torch.where(delta<=1e-6,(lower<threshold).to(risk.dtype),torch.where(threshold<=lower,torch.zeros_like(risk),torch.where(threshold>=upper,torch.ones_like(risk),risk)))
        return {'prediction':mean,'lower':quantile(.05),'upper':quantile(.95),'mixture_pi':pi,'mixture_mu':mu,'mixture_sigma':sigma,'bound_lower':lower,'bound_upper':upper,'reliability':q.squeeze(-1),'samples':flat,'sample_weight':mass,'low_yield_probability':risk,'prefix_length':torch.full((n,1),k,device=x.device)}
    def forward(self,b):
        total=b['x'].shape[1];fraction=self.c.get('prefix_fraction',.25)
        if self.training and self.c['variant']!='single_prefix':fraction=float(torch.tensor([.1,.25,.5,.75,1.])[torch.randint(5,())])
        return self.predict_at(b,max(1,int(total*fraction)))
    def objective(self,b,p):
        y=b['y'].float();delta=p['bound_upper']-p['bound_lower'];coordinate=((y-p['bound_lower'])/delta.clamp_min(1e-6)).clamp(1e-5,1-1e-5);z=torch.logit(coordinate)
        log_density=-.5*((z[:,None]-p['mixture_mu'])/p['mixture_sigma']).square()-p['mixture_sigma'].log()-.918939
        nll=-torch.logsumexp(p['mixture_pi'].clamp_min(1e-8).log()+log_density,-1)+(delta.clamp_min(1e-6)*coordinate*(1-coordinate)).log();nll=torch.where(delta>1e-6,nll,torch.zeros_like(nll))
        sample=p['samples'];weight=p['sample_weight'];crps=(weight*abs(sample-y[:,None])).sum(-1)-.5*(weight[:,:,None]*weight[:,None,:]*abs(sample[:,:,None]-sample[:,None,:])).sum((1,2))
        loss=nll.mean()+.25*crps.mean()
        if self.c['variant']!='no_reliability':
            loss=loss+.1*F.binary_cross_entropy(p['reliability'],torch.ones_like(p['reliability']))
            corrupted=dict(b);values=b['x'].clone();values[:,0,:,:2]=values[:,0,:,:2]+4;corrupted['x']=values
            corrupt=self.predict_at(corrupted,int(p['prefix_length'][0,0]));target=torch.ones_like(corrupt['reliability']);target[:,0]=0
            loss=loss+.1*F.binary_cross_entropy(corrupt['reliability'],target)
        length=int(p['prefix_length'][0,0])
        if length<b['x'].shape[1] and self.c['variant']!='baseline':
            following=self.predict_at(b,length+1);change=following['upper']-following['lower']-p['upper']+p['lower']
            loss=loss+.2*(following['reliability'][:,-1]*F.relu(change)).mean()
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(n,8,16,4)).astype('float32');x[...,2]=(rng.random((n,8,16))>.2);x[...,3]=1
    return {'x':x,'planned_dies':np.full(n,128,np.float32),'y':x[...,2].mean((1,2)),'group':np.arange(n)}


def measure(p,d,c):
    sample=p['samples'];weight=p['sample_weight'];y=d['y'];crps=(weight*abs(sample-y[:,None])).sum(-1)-.5*(weight[:,:,None]*weight[:,None,:]*abs(sample[:,:,None]-sample[:,None,:])).sum((1,2))
    return {'crps':float(crps.mean()),'low_yield_brier':float(np.mean((p['low_yield_probability']-(y<c.get('low_yield',.7)))**2)),'plan_bound_violation':float(np.mean((p['prediction']<p['bound_lower'])|(p['prediction']>p['bound_upper'])))}
