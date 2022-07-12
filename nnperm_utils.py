import collections
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np


def error_barrier_from_losses(errors, reduction='none'):
    n_samples = errors.shape[0]
    alphas = error_barrier_linspace_sample(n_samples)
    error_barriers = [e - (a * errors[0] + (1 - a) * errors[-1]) for e, a in zip(errors, alphas)]
    error_barriers = np.stack(error_barriers, axis=0)
    if reduction == 'mean':
        error_barriers = np.mean(error_barriers, axis=1)
    if reduction == 'sum':
        error_barriers = np.sum(error_barriers, axis=1)
    return error_barriers

def evaluate_per_sample(model, dataloader, state_dict=None, loss_fn=None, device="cuda"):
    if state_dict is not None:
        model = deepcopy(model)
        model.load_state_dict(deepcopy(state_dict))
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_examples, batch_labels in dataloader:
            batch_examples = batch_examples.to(device=device)
            y = model(batch_examples)
            if loss_fn is not None:
                y = loss_fn(y, batch_labels.to(device=device))
            outputs.append(y.cpu().detach().numpy())
    return np.concatenate(outputs)

def calculate_errors(model, model_state_dict1, model_state_dict2, dataloader, n_samples=10):
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    errors = []
    for alpha in tqdm(error_barrier_linspace_sample(n_samples)[1:-1]):
        avg_weight = collections.OrderedDict()

        for k in model_state_dict1.keys():
            avg_weight[k] = alpha*model_state_dict1[k].clone() + (1-alpha)*model_state_dict2[k].clone()

        errors.append(evaluate_per_sample(model, dataloader, state_dict=avg_weight, loss_fn=ce_loss))
    error1 = evaluate_per_sample(model, dataloader, state_dict=model_state_dict1, loss_fn=ce_loss)
    error2 = evaluate_per_sample(model, dataloader, state_dict=model_state_dict2, loss_fn=ce_loss)
    errors = np.stack([error1] + errors + [error2], axis=0)
    return errors

def error_barrier_linspace_sample(n_samples):
    return np.linspace(0., 1., n_samples)
