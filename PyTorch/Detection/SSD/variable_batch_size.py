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
  def __init__(self, optimizer, log_steps=None, reduction='mean', start_epoch=0):
    self.optimizer = optimizer
    self.set_epoch(start_epoch)
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
      all_reduce_tensors(grads, scale)
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