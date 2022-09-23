import torch.distributed as dist
import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
try:
  import torch_xla.core.xla_model as xm
except:
  print("    Running without torch_xla")

class CustomSampler(DistributedSampler):
  """
  Returns a batch of indices at a time.

  Intricacies:
    No step of the trace gets dropped.
    But data at the end of the epoch might get dropped.
  """
  def __init__(self, dataset, trace, seed=42, num_replicas=None, rank=None, 
    minibatch_size=32, verbose=True):
    """
    Loads the trace from a CSV file.

    dataset: Dataset used for sampling
    trace: path or list or int
    seed (optional): Random seed to use for shuffling
    num_replicas (optional): Number of processes participating in
        distributed training.
    rank (optional): Rank of the current process within num_replicas.
    start_index (optional):  Which index of the permuted dataset to start sampling form, it should be iteration_id * global_batch_size
    batch_size (optional): Sample per consecutive batches of this size (per GPU). If set to 1, it corresponds to the native Distributed Sampler
    """

    super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
    self.seed = seed
    self.minibatch_size = minibatch_size
    self.verbose = verbose
    
    # Load trace
    if isinstance(trace, str):
      df = pd.read_csv(trace)
      self.trace = df.batchsize.to_list()
    elif isinstance(trace, list):
      self.trace = trace
    elif isinstance(trace, int):
      self.trace = [trace] * 2000000
    if verbose and rank==0:
        print(f"    MySampler (rank {rank}): Loaded Trace {trace} starting with {self.trace[0]}, {self.trace[1]},...")

    # Check trace is valid
    tot_samples = 0
    for s in self.trace:
        if s%minibatch_size != 0:
            raise ValueError(f"Every Sample batch size should be multiple of {minibatch_size}")
        tot_samples += s

    # Check length of trace
    self.minibatches_per_epoch = len(self.dataset) // minibatch_size
    self.max_epoch = tot_samples // (self.minibatches_per_epoch * minibatch_size)
    if verbose and rank==0:
        print(f"    MySampler (rank {rank}): trace_samples={tot_samples}, minibatches_per_epoch={self.minibatches_per_epoch} => max_epoch=~{self.max_epoch}")

  def __len__(self) -> int:
    return len(self.__iter__())

  def set_epoch(self, epoch: int) -> None:
    if (epoch > self.max_epoch):
      print(f"    Custom Sampler Warning: epoch > max_epoch")
    self.epoch = epoch

  def get_batch_assignments(self, epochtrace):
    """
    @parameter epochtrace: Trace of batchsizes for this epoch. 
      Sum should be smaller equal than minibatches_per_epoch*minibatch_size
    Returns a list of lists where sublist j contains the minibatche-indices contributing to step j from this rank.
    A sublist with more than one element implies accumulation.
    """
    batches = []
    step_start = 0
    all_minibatches = list(range(self.minibatches_per_epoch))
    steps_per_epoch = len(epochtrace)
    for i in range(steps_per_epoch):
      minibatches_per_step = epochtrace[i] // self.minibatch_size
      step_end = step_start + minibatches_per_step
      local_minibatches = all_minibatches[(step_start + self.rank) : step_end : self.num_replicas]
      batches.append(local_minibatches)
      step_start = step_end

    return batches

  def get_epoch_range(self, epoch):
    start, end, epochsum, e = 0, 0, 0, -1
    # INVARIANT: [start..end-1] is the partition for epoch *epoch*
    while (e < epoch):
      start = end
      while (epochsum + self.trace[end] <= self.minibatch_size*self.minibatches_per_epoch):
        epochsum += self.trace[end]
        end += 1
      e += 1
      epochsum = 0
    return start, end

  def get_epochtrace(self, epoch):
    """
    Returns the list of batchsizes for this epoch
    """
    start, end = self.get_epoch_range(epoch)
    trace = self.trace[start:end]
    assert(sum(trace) <= self.minibatch_size*self.minibatches_per_epoch)
    return trace

  def __iter__(self):
    """
    Returns a subset of [0, len(dataset)-1] partitioned into minibatches. Every index occurs at most once, except index 0, 
    which is used when the core doesn't actually participate in the step.
    The set of iterators for all ranks forms a partition of [0, len(dataset)-1]. 
    The sum of globally seen minibatches over a step including accumulation is equal 
    to the desired global batch size at step i.

    Make sure self.epoch is set correctly. 
    """
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    # first permute, and then select
    indices = torch.randperm(len(self.dataset), generator=g).tolist()
    indices = indices[:self.minibatches_per_epoch*self.minibatch_size]

    # Batchsizes for each step of the epoch
    epochtrace = self.get_epochtrace(self.epoch)
    # Batches assigned to this rank for each step
    batchlist = self.get_batch_assignments(epochtrace)
    # Batches to concrete indices
    steplist = []
    for batches in batchlist:
      if (batches != []):
        for bi in batches:
          steplist.append(indices[bi*self.minibatch_size:(bi+1)*self.minibatch_size])
      else:
        steplist.append([0 for _ in range(self.minibatch_size)])

    #if self.verbose:
      #print(f"Batch Assignment for first step worker {self.rank}: {batchlist[0]}")
    return iter(steplist)


class CustomOptimizer(object):
  def __init__(self, optimizer, sampler, log_steps=None, reduction='mean'):
    self.optimizer = optimizer
    self.sampler = sampler
    self.set_epoch(0)
    self.log_steps = log_steps
    self.reduction = reduction
    if (reduction != 'mean'):
      xm.master_print(f"Using {reduction} reduction!")

  def set_epoch(self, epoch):
    self.epochtrace = self.sampler.get_epochtrace(epoch)
    self.divisors = [t // self.sampler.minibatch_size for t in self.epochtrace]
    self.schedule = self.sampler.get_batch_assignments(self.epochtrace)
    self.step_index = 0
    self.substep_index = 0
    self.optimizer.zero_grad()

  def step(self):
    """
    Returns true if self.optimizer.step was called
    """
    subbatches = self.schedule[self.step_index]
    # Ignore dummy batches
    if subbatches == []:
      self.optimizer.zero_grad()
      #if self.verbose: 
        #print(f"    CustomOptimizer (rank {self.sampler.rank}): Empty Step")
    # Tyme to sync
    if subbatches == [] or self.substep_index == len(subbatches)-1:
      grads = self._fetch_gradients()
      scale = 1.0 if self.reduction=='sum' else 1.0/self.divisors[self.step_index]
      #all_reduce_tensors(grads, scale)
      scale_gradients(self.optimizer, scale)
      # Adjust LR
      ref_lr = self.get_lr()
      self.adapt_lr()
      self.optimizer.step()
      if self.log_steps is not None and (self.step_index % self.log_steps == 0): 
        xm.master_print(f"    CustomOptimizer: Step={self.step_index}, bs={self.epochtrace[self.step_index]}, lr={self.get_lr()}")

      self.optimizer.zero_grad()
      self.set_lr(ref_lr)
      self.step_index += 1
      self.substep_index = 0
      xm.mark_step()
      return True
    else:
      self.substep_index += 1
      return False

  def get_lr(self):
    values = tuple(param_group['lr'] for param_group in self.optimizer.param_groups)
    return values

  def set_lr(self, values):
    for i, data in enumerate(zip(self.optimizer.param_groups, values)):
      param_group, lr = data
      param_group['lr'] = lr

  def get_bs(self):
    """
    Returns batchsize at current step_index, 0 if past epoch.
    """
    if self.step_index < len(self.epochtrace):
      return self.epochtrace[self.step_index]
    else: return 0

  def adapt_lr(self, batchsize):
    pass

  def state_dict(self):
    pass

  def load_state_dict(self):
    pass

  def zero_grad(self):
    raise Exception

  def _fetch_gradients(self):
    """
    Provides list of gradient tensors.
    """
    gradients = []
    for param_group in self.optimizer.__getstate__()['param_groups']:
      for group, params in param_group.items():
        if group == 'params':
          for p in params:
            if isinstance(p, torch.Tensor) and p.grad is not None:
              gradients.append(p.grad.data)
    return gradients


class LinearRuleOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(LinearRuleOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def F(ref_lr, batchsize, ref_batchsize):
    return batchsize/ref_batchsize * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(LinearRuleOptimizer.F(rlr, bs, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return LinearRuleOptimizer.F(ref_lr, bs, self.ref_batchsize)


class ConstantOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(ConstantOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def F(ref_lr, batchsize, ref_batchsize):
    return ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(ConstantOptimizer.F(rlr, bs, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return ConstantOptimizer.F(ref_lr, bs, self.ref_batchsize)

class LinearRuleMomentumOptimizer(CustomOptimizer):
  """
  The gradients across subbatches are no longer averaged but summed over subbatches instead.
  The learning rate is the linear scaled learning rate for a subbatch of 32.
  As the scale of the gradients changes, the scale of the weight decay must also!
  Only for single Parameter Group, very coupled."""
  def __init__(self, optimizer, sampler, ref_batchsize, weight_decay=1e-4, log_steps=None):
    super(LinearRuleMomentumOptimizer, self).__init__(optimizer, sampler, log_steps, reduction='sum')
    self.ref_batchsize = ref_batchsize
    self.weight_decay=weight_decay

  def F(ref_lr, batchsize, ref_batchsize):
    return batchsize/ref_batchsize * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(LinearRuleMomentumOptimizer.F(rlr, self.sampler.minibatch_size, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)
    # TRICKY: Need to correct weight decay!!
    for group in self.optimizer.param_groups:
      group['weight_decay'] = self.weight_decay*self.divisors[self.step_index]

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return LinearRuleMomentumOptimizer.F(ref_lr, bs, self.ref_batchsize)


class RootRuleOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(RootRuleOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def F(ref_lr, batchsize, ref_batchsize):
    return (batchsize/ref_batchsize)**0.5 * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(RootRuleOptimizer.F(rlr, bs, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return RootRuleOptimizer.F(ref_lr, bs, self.ref_batchsize)


class AdaScaleOptimizer(CustomOptimizer):
  """
  Implements the version of section 3.4 of the paper.
  As the AdaScale gain "influences" the current step, a special iteration loop 
  is required, i.e. step returns the gain.
  Args:
    optimizer (torch.optimizer): e.g. SGD
    sampler (CustomSampler): The custom sampler used.
    log_steps (int): Log Interval
  """
  def __init__(self, optimizer, sampler, log_steps=None, ref_batchsize=32):
    super(AdaScaleOptimizer, self).__init__(optimizer, sampler, log_steps)
    # The norm of the per-minibatch gradients get accumulated here
    self.accum_grad_sqr = np.zeros(len(self.optimizer.param_groups))
    # Exponential Moving Averages
    self.grad_sqr_batchavg_avg = None
    self.grad_sqr_avg = None
    # Required to interpret learning rates from lr_schedule
    self.ref_batchsize = ref_batchsize
    self.last_grads = [[torch.zeros_like(grad) for grad in grad_group] for grad_group in self._fetch_gradients_grouped()]
  
  def adapt_lr(self):
    """
    Called when gradients across minibatches have been averaged.
    Updates the exponential averages and sets the learning rate accordingly.
    """
    # Compute average minigradient norm_sqr over total batch
    self.accum_grad_sqr = all_reduce_tensors_mesh(tag="accumulated_grad_norms", data=self.accum_grad_sqr, 
      scale=1.0/self.divisors[self.step_index])

    # Compute gradient norm_sqr
    xm.mark_step()
    grad_sqr = np.zeros(len(self.optimizer.param_groups)) 
    for (i, grad_group) in enumerate(self._fetch_gradients_grouped()):
      for grad in grad_group:
        grad_sqr[i] += grad.pow(2).sum().item()

    # Update exponential moving averages
    theta = max(0.0, 1-self.divisors[self.step_index]/1000)
    if (self.grad_sqr_avg is not None):
      self.grad_sqr_batchavg_avg = (1-theta) * self.accum_grad_sqr + theta * self.grad_sqr_batchavg_avg
      self.grad_sqr_avg = (1-theta) * grad_sqr + theta * self.grad_sqr_avg
    else:
      self.grad_sqr_batchavg_avg = self.accum_grad_sqr.copy()
      self.grad_sqr_avg = grad_sqr

    # Update lr
    lr = self.current_lr()
    self.set_lr(lr)
    # Reset for next large step
    self.accum_grad_sqr.fill(0.0)

  def gain(self):
    """Gain, tuple, entry for each param group"""
    # Check for numerical imprecision
    if np.any(self.grad_sqr_avg < 1e-5):
      print("WARNING: possible numerical imprecision")

    m1 = np.clip(self.grad_sqr_batchavg_avg, a_min=0.0, a_max=None)
    m2 = np.clip(self.grad_sqr_avg, a_min=1e-6, a_max=None)
    return tuple((m1/m2).tolist())

  def current_lr(self):
    lr = [ref_lr / self.ref_batchsize * self.sampler.minibatch_size * gain for (ref_lr, gain) 
      in zip(self.get_lr(), self.gain())]
    return tuple(lr)

  def step(self):
    """ For each minibatch, keep track of the norm of the individual gradient. 
        As gradients get accumulated, take differences to measure impact of individual minibatch."""
    # List of gradient list for each parameter group
    updated_grads = self._fetch_gradients_grouped()

    # Record norm of diff
    xm.mark_step()
    with torch.no_grad():
      for i in range(len(updated_grads)):
        local_grad_sqr = torch.tensor(0.0, device=xm.xla_device())
        for (last_grad, updated_grad) in zip(self.last_grads[i], updated_grads[i]):
          local_grad_sqr += (updated_grad-last_grad).pow(2).sum()

        self.accum_grad_sqr[i] += local_grad_sqr.item()
        self.last_grads[i] = [grad.detach().clone() for grad in updated_grads[i]] 

    # Perform normal step
    return super().step()

  def _fetch_gradients_grouped(self):
    """
    Provides list of gradient tensors.
    """
    gradient_groups = []
    for param_group in self.optimizer.__getstate__()['param_groups']:
      gradients = []
      for group, params in param_group.items():
        if group == 'params':
          for p in params:
            if isinstance(p, torch.Tensor) and p.grad is not None:
              gradients.append(p.grad.data)
      gradient_groups.append(gradients)
    return gradient_groups


class AdaScaleOptimizer2(CustomOptimizer):
  """
  Implements the version of Appendix B.3 of the paper.
  As the AdaScale gain "influences" the current step, a special iteration loop 
  is required, i.e. step returns the gain.
  Args:
    optimizer (torch.optimizer): e.g. SGD
    sampler (CustomSampler): The custom sampler used.
    log_steps (int): Log Interval
  """
  def __init__(self, optimizer, sampler, log_steps=None, ref_batchsize=32):
    super(AdaScaleOptimizer2, self).__init__(optimizer, sampler, log_steps)
    # The norm of the per-minibatch gradients get accumulated here
    self.accum_grad_sqr = np.zeros(1)
    # Exponential Moving Averages
    self.grad_sqr_avg = np.ones(1)
    self.grad_var_avg = np.zeros(1)
    # Required to interpret learning rates from lr_schedule
    self.ref_batchsize = ref_batchsize
    self.last_grads = None
    self._scale = 1.0
    xm.master_print(f"Model has {len(self.optimizer.param_groups)} parameter groups!")
  
  def adapt_lr(self):
    """
    Called when gradients across minibatches have been averaged.
    Updates the exponential averages and sets the learning rate accordingly.
    """
    # Scale (S)
    scale = self.divisors[self.step_index]
    if (self._scale != scale):
      self.grad_var_avg *= self._scale / scale
      self._scale = scale

    # Compute sum minigradient norm_sqr over total batch
    #self.accum_grad_sqr = all_reduce_tensors_mesh(tag="accumulated_grad_norms", data=self.accum_grad_sqr, 
    #  scale=1.0)

    # Compute gradient norm_sqr
    xm.mark_step()
    grad_norm_sqr = np.zeros(1) 
    for grad in self._fetch_gradients():
      grad_norm_sqr[0] += grad.pow(2).sum().item()

    # Estimate grad_sqr, grad_var
    grad_var = (1.0/(scale-1.0)) * self.accum_grad_sqr - (scale/(scale-1.0)) * grad_norm_sqr
    grad_sqr = grad_norm_sqr - (1.0/scale) * grad_var
    grad_var = np.clip(grad_var, a_min=1e-6, a_max=None)
    grad_sqr = np.clip(grad_sqr, a_min=0.0, a_max=None)

    # Update exponential moving averages
    theta = max(0.0, 1-scale/1000)
    self.grad_var_avg = (1-theta) * grad_var + theta * self.grad_var_avg
    self.grad_sqr_avg = (1-theta) * grad_sqr + theta * self.grad_sqr_avg

    # Update lr
    lr = self.current_lr()
    self.set_lr(lr)

    # Reset for next large step
    self.accum_grad_sqr.fill(0.0)

    # Log
    if self.log_steps is not None and (self.step_index % self.log_steps == 0): 
      xm.master_print(f"    Adascale: Step={self.step_index}, gain={self.gain()}, grad_var_avg={self.grad_var_avg}, grad_sqr_avg={self.grad_sqr_avg}")

  def gain(self):
    """Gain"""
    var = self.grad_var_avg
    sqr = self.grad_sqr_avg
    gain = (var + sqr) / (var / self._scale + sqr)
    return gain.item()

  def current_lr(self):
    gain_singleton = self.gain()
    ref_lr = self.get_lr()[0]
    lr = ref_lr / self.ref_batchsize * self.sampler.minibatch_size * gain_singleton
    return tuple([lr for _ in self.optimizer.param_groups])

  def step(self):
    """ For each minibatch, keep track of the norm of the individual gradient. 
        As gradients get accumulated, take differences to measure impact of individual minibatch."""
    # List of gradient list for each parameter group
    updated_grads = self._fetch_gradients()

    # First step
    if self.last_grads is None:
      self.last_grads = [torch.zeros_like(grad, requires_grad=False) for grad in updated_grads]

    # Record norm of diff
    xm.mark_step()
    with torch.no_grad():
        local_grad_sqr = torch.tensor(0.0, device=xm.xla_device())
        for (last_grad, updated_grad) in zip(self.last_grads, updated_grads):
          local_grad_sqr += (updated_grad-last_grad).pow(2).sum()

        self.accum_grad_sqr[0] += local_grad_sqr.item()
        for j in range(len(self.last_grads)):
          self.last_grads[j].copy_(updated_grads[j])

    # Perform normal step
    return super().step()


def init_group(local_ordinal, cores_per_host=8):
  """
  Initialize Group
  @return global ordinal, global world size
  """ 
  host_ordinal = int(os.environ.get('MY_HOST_ORDINAL', default="0"))
  host_world_size = int(os.environ.get('MY_HOST_WORLD_SIZE', default="1"))
  global_ordinal = cores_per_host*host_ordinal + local_ordinal
  global_world_size = host_world_size * cores_per_host

  if (host_world_size > 1 and local_ordinal == 0):
    init_method = 'tcp://' + os.environ.get('MY_DIST_ROOT')
    dist.init_process_group('gloo', init_method=init_method, rank=host_ordinal, world_size=host_world_size)
    print(f"    ---- Inter Node Process Group Initialized ----")

  return (global_ordinal, global_world_size)


def all_reduce_tensors(tensors, scale):
  """
  Perform global two-level all reduce.
  Args:
    tensors: List of `torch.Tensor`
    scale (float): scaling factor
  """
  # Locally Reduce
  xm.all_reduce(reduce_type=xm.REDUCE_SUM, inputs=tensors, scale=scale)
  
  # Inter-Node Reduce
  host_world_size = int(os.environ.get('MY_HOST_WORLD_SIZE', default="1"))
  if host_world_size > 1:
    if xm.get_ordinal() == 0:
      xm.mark_step()
      cpu_tensors = [tensor.cpu() for tensor in tensors]
      dist.all_reduce_coalesced(cpu_tensors, op=dist.ReduceOp.SUM)
      for i in range(len(tensors)):
        tensors[i].copy_(cpu_tensors[i])
      
    # Broadcast to other cores
    # Other cores don't contribute
    if xm.get_ordinal() != 0:
      for tensor in tensors:
        tensor.fill_(0.0)
    xm.all_reduce(xm.REDUCE_SUM, tensors)


def all_reduce_tensors_mesh(tag, data, scale=1.0):
  """
  Perform global two-level all reduce.
  Args:
    data: List of `torch.Tensor`
    scale (float): scaling factor
  """
  # Inter-Node Reduce
  host_world_size = int(os.environ.get('MY_HOST_WORLD_SIZE', default="1"))
  if host_world_size > 1:
    raise RuntimeError

  # Locally Reduce
  reduce_fn = lambda x: np.sum(x, axis=0)
  x = xm.mesh_reduce('m', data, reduce_fn)
  return scale * x

def scale_gradients(optimizer, scale):
    for param_group in optimizer.__getstate__()['param_groups']:
      for group, params in param_group.items():
        if group == 'params':
          for p in params:
            if isinstance(p, torch.Tensor) and p.grad is not None:
                p.grad.mul_(scale)
