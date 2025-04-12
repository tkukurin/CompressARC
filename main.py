import os
import time
import pickle
import tqdm
import argparse
import fire

import numpy as np
import torch
import matplotlib.pyplot as plt

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization

# Set initial random seeds
np.random.seed(0)
torch.manual_seed(0)

def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    """
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def take_step(task, model, optimizer, train_step, train_history_logger):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """
    optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

    # Compute the total KL loss
    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)

    # Compute the reconstruction error
    reconstruction_error = 0
    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            # Determine whether the grid size is already known.
            # If not, there is an extra term in the reconstruction error, corresponding to
            # the probability of reconstructing the correct grid size.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            coefficient = (
                0.01**max(0, 1-train_step/100)
                if grid_size_uncertain else 1)
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            # Given that we have the correct grid size, get the reconstruction error of getting the colors right
            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition  # given the correct grid size,
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  # calculate the error for the colors.
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient  # Aggregate for all possible grid sizes
            reconstruction_error = reconstruction_error - logprob

    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_history_logger.log(
        train_step, logits, x_mask, y_mask, KL_amounts, 
        KL_names, total_KL, reconstruction_error, loss)

def post_mortem_debug(enable_traceback=True):
    """Call and enter pdb after an exception occurs globally."""
    import traceback; import pdb; import sys
    def excepthook(exc_type, exc_value, exc_traceback):
        if enable_traceback:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("\nEntering post-mortem debugging...\n")
        pdb.post_mortem(exc_traceback)
    sys.excepthook = excepthook

def save_training_results(folder, task_name, train_history_logger, model):
    """Save the metrics, model weights, and learned representations."""
    np.savez(folder + task_name + '_KL_curves.npz',
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

def load_training_results(folder, task_name):
    """Load saved training results and model data."""
    stored_data = np.load(folder + task_name + '_KL_curves.npz', allow_pickle=True)
    return {
        'KL_curves': stored_data['KL_curves'][()],
        'reconstruction_error_curve': stored_data['reconstruction_error_curve'],
        'multiposteriors': stored_data['multiposteriors'][()],
        'target_capacities': stored_data['target_capacities'][()],
        'decode_weights': stored_data['decode_weights'][()]
    }

def plot_KL_components(folder, task_name, KL_curves):
    """Plot the KL curves over time with special coloring for specified tasks."""
    special_curve_colors = {
            '272f95fa': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (0,1,1,0,0), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            },
            '6cdd2623': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,1,0,0,0), (1,0,0,1,1), (0,0,1,0,0)],
                'colors': [(1, 0.6, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1), (1, 0, 0.5)]
            },
            '41e4d17e': {
                'dims': [(1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 0, 1)]
            },
            '6d75e8bb': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            }
    }
    
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        line_color = (0.5, 0.5, 0.5)
        label = None
        if task_name in special_curve_colors:
            dims_list = special_curve_colors[task_name]['dims']
            colors_list = special_curve_colors[task_name]['colors']
            for dims, color in zip(dims_list, colors_list):
                if tuple(eval(component_name)) == dims:
                    line_color = color
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    axis_names = [axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
                    label = '(' + ', '.join(axis_names) + ', channel)'
        ax.plot(np.arange(curve.shape[0]), curve, color=line_color, label=label)
    if task_name == '6cdd2623':
        ax.set_ylim((0.3, 4e4))

    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_components.png', bbox_inches='tight')
    plt.close()

def plot_KL_vs_reconstruction(folder, task_name, KL_curves, reconstruction_error_curve):
    """Plot the KL vs reconstruction error over time."""
    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]),
            reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()

def get_averaged_samples(multiposteriors, target_capacities, decode_weights, n_samples=100):
    """Get averaged samples from the model."""
    samples = []
    for i in range(n_samples):
        sample, KL_amounts, KL_names = layers.decode_latents(target_capacities,
                                       decode_weights, multiposteriors)
        samples.append(sample)

    def average_samples(dims, *items):
        mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
        all_but_last_dim = tuple(range(len(mean.shape) - 1))
        mean = mean - np.mean(mean, axis=all_but_last_dim)
        return mean
    
    return multitensor_systems.multify(average_samples)(*samples), KL_amounts, KL_names

def visualize_learned_representations(folder, task_name, task, results):
    """Visualize the learned tensor representations and their principal components."""
    means, KL_amounts, KL_names = get_averaged_samples(
        results['multiposteriors'], 
        results['target_capacities'], 
        results['decode_weights']
    )
    
    # Figure out which tensors contain significant information
    dims_to_plot = []
    for KL_amount, KL_name in zip(KL_amounts, KL_names):
        dims = tuple(eval(KL_name))
        if torch.sum(KL_amount).detach().cpu().numpy() > 1:
            dims_to_plot.append(dims)
    
    # Show the top principal components of the significant tensors
    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
    restricted_color_names = [color_names[i] for i in task.colors]
    restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist())
                              for i in task.colors]
    
    for dims in dims_to_plot:
        tensor = means[dims]
        
        orig_shape = tensor.shape
        if len(orig_shape) == 2:
            tensor = tensor[None,:,:]
        orig_shape = tensor.shape
        if len(orig_shape) == 3:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]  # Calculate component strength

                # Show the component
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)
                
                # Pick the axis labels
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                # Set the tick labels
                # Tick labels for example axis
                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                # Tick labels for color axis
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                # Tick labels for direction axis
                direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                # Standard tick labels for height and width axes

                ax.set_title('component' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()

        # Plot an ({example, color, direction}, x, y) tensor with subplots
        elif len(orig_shape) == 4 and dims[3] == 1 and dims[4] == 1:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get the top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                n_plots = orig_shape[0]

                # Make the subplots
                fig, axs = plt.subplots(1, n_plots)
                for plot_idx in range(n_plots):
                    ax = axs[plot_idx] if n_plots > 1 else axs
                    ax.imshow(component[plot_idx,:,:], cmap='gray', vmin=-1, vmax=1)

                    # Get the axis labels
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    tensor_name = '_'.join([axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                    ax_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][2]
                    ax.set_ylabel(x_dim)
                    ax.set_xlabel(y_dim)

                    # Standard tick labels for height and width axes

                    # Label the subplots
                    if ax_dim == 'example':
                        ax.set_title('example ' + str(plot_idx))
                    elif ax_dim == 'color':
                        ax.set_title(restricted_color_names[plot_idx],
                                     color=restricted_color_codes[plot_idx],
                                     fontweight="bold")
                    elif ax_dim == 'direction':
                        direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                        ax.set_title(direction_names[plot_idx], fontsize=22)

                plt.subplots_adjust(wspace=1)
                fig.suptitle('component ' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.subplots_adjust(top=1.4)
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()

def setup_device(backend='cpu'):
    """Set up the computation device."""
    if torch.backends.mps.is_available() and backend == 'mps':
        print("[WARN] trains _slower_ on MPS, probs due to CPU fallback")
        import warnings; warnings.filterwarnings("ignore", message=".*_cummax_helper.*not currently supported.*")
        import os; assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1", (
            "PLS RUN WITH:\n$ PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py"
            "\nOr suffer 'NotImplementedError: operator 'aten::_cummax_helper' (...) MPS. "
            "Enable CPU fallback for operations not supported on MPS'"
        )
        torch.set_default_device("mps")
        torch.set_default_dtype(torch.float32)  # req for mps!
    elif backend == 'cuda' and torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")
    
    print(f"{torch.get_default_dtype()=}", f"{torch.get_default_device()=}", sep="\n")


def get_task_split(split):
    task_names = [
        "arc-agi_training", "arc-agi_evaluation", "arc-agi_test",
        "arc-agi2_training", "arc-agi2_evaluation"]
    
    if isinstance(split, int):
        return task_names[split]
    return split


def train_or_analyze(
    spec: str = "arc-agi2_training:0",
    debug: bool = False,
    backend: str = 'cpu', iterations: int = 2000, plot_interval: int|None = 50
):
    if debug: post_mortem_debug()
    setup_device(backend)
    
    split, parts = spec.split(":")
    split_name = get_task_split(split)
    safeint = lambda x: int(x) if x.isdigit() else x
    tasks = [safeint(x) for x in parts.split(",")]
    tasks = preprocessing.preprocess_tasks(split_name, tasks or None)
    os.makedirs((folder := f'results/{split_name}/'), exist_ok=True)
    
    models = []; optimizers = []; train_history_loggers = []
    for task in tqdm.tqdm(tasks, desc=f"Tasks ({split_name}:{len(tasks)})"):
        models.append(model := arc_compressor.ARCCompressor(task))
        optimizers.append(torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9)))
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    true_solution_hashes = [task.solution_hash for task in tasks]
    to_train = list(zip(tasks, models, optimizers, train_history_loggers))
    for i, (task, model, optimizer, train_history_logger) in tqdm.tqdm(enumerate(to_train), desc="Training"):
        task_name = task.task_name
        for train_step in tqdm.trange(iterations, desc="Training steps", leave=False):
            take_step(task, model, optimizer, train_step, train_history_logger)
            if plot_interval is not None and (train_step+1) % plot_interval == 0:
                visualization.plot_solution(train_history_logger,
                    fname=f"{folder}{task_name}_at_{train_step+1}_steps.png")
        
        visualization.plot_solution(train_history_logger)
        if plot_interval is not None:
            save_training_results(folder, task_name, train_history_logger, model)
            results = load_training_results(folder, task_name)
            plot_KL_components(folder, task_name, results['KL_curves'])
            plot_KL_vs_reconstruction(
                folder, task_name, results['KL_curves'],
                results['reconstruction_error_curve'])
            visualize_learned_representations(folder, task_name, task, results)
        else:
            solution_selection.save_predictions(
                train_history_loggers[:i+1], fname=os.path.join(folder, 'preds.npz'))
            solution_selection.plot_accuracy(
                true_solution_hashes, fname=os.path.join(folder, 'acc.npz'))


if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(train_or_analyze)
    print("Done", (t := time.time() - start_time))
    with open('timing_result.txt', 'w') as f: f.write(f"dt (sec): {t}")