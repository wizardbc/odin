import torch
import torch.nn.functional as F
import functorch

from metrics import runner

@runner
class ODIN:
  def __init__(self, model, epsilon=0.002, T=1000, device='cuda:1'):
    self.epsilon = epsilon
    self.T = T
    self.device = device
    
    model.eval()
    model.to(device)
    self.model = model

    # for input processing
    def logS(img) -> float:
      img = img.unsqueeze(0)
      logit = model(img)/T
      target = torch.argmax(logit, dim=-1)
      return F.cross_entropy(logit, target)
    self.logS_grad = functorch.vmap(functorch.grad(logS), in_dims=(0,))

  @torch.no_grad()
  def forward(self, imgs) -> torch.Tensor:
    imgs = imgs.to(self.device)

    if self.epsilon == .0:
      return F.softmax(self.model(imgs)/self.T, dim=-1)
    
    noise = self.logS_grad(imgs)
    noise = (torch.ge(noise, 0).float() - 0.5) * 2
    std = torch.tensor((63.0/255, 62.1/255.0, 66.7/255.0)).reshape(1,3,1,1).to(self.device)
    noise = noise / std
    imgs = torch.add(imgs, noise, alpha=-self.epsilon)
    return F.softmax(self.model(imgs)/self.T, dim=-1)
  
  __call__ = forward