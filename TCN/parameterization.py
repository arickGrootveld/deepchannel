import numpy as np
from model import TCN

# Full credit to Wasi Ahmad for posting this snippet to stackoverflow
# The original snippet can be found at https://stackoverflow.com/questions/48393608/pytorch-network-parameter-calculation
def count_parameters_tcn(tcn_model):
    total_param = 0
    for name, param in tcn_model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

# If this code is called directly, assume it is being used to read the parameters 
# from a network
if __name__ == "__main__":
    import argparse
    import torch as t
    parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
    parser.add_argument('--model_path', type=str, default='None',
                    help='The location to load the model from (default=None)')



    args = parser.parse_args()

    modelPath = args.model_path

    print("loading model from: " + modelPath)
    
    modelContext = t.load(modelPath)
    
    # If the file is one that was generated from our code then load it our way
    if "model_parameters" in modelContext:
        inter = modelContext['model_parameters']
        inter['state_dict'] = modelContext['model_state_dict']
        modelContext = inter

    model = TCN(modelContext['input_channels'], modelContext['n_classes'], modelContext['channel_sizes'], kernel_size=modelContext['kernel_size'], dropout=modelContext['dropout'])

    model.load_state_dict(modelContext['state_dict'])
    
    print('number of trainable parameters =', count_parameters_tcn(model))



#def count_parameters_tcn(tcn_model):
#    total_param = 0
#    for name, param in tcn_model.named_parameters():
#        if param.requires_grad:
#            num_param = np.prod(param.size())
#            if param.dim() > 1:
#                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
#            else:
#                print(name, ':', num_param)
#            total_param += num_param
#    return total_param
