def generate_filters(conv_layers_nb, filters_base, begin_factor=2):
    filters = [begin_factor*filters_base]
    interv = 2
    difference = 1
    factor = begin_factor
    for i in range(conv_layers_nb-1):
        if (i+1)%interv == 0 and i != 0:
            difference *= 2
        factor += difference
        filters.append(factor*filters_base)
    return filters

def generate_logname(model_params):
    logname = ''
    logname += 'fnb_' + str(len(model_params['filters'])) + '--'
    logname += 'fbase_' + str(int(model_params['filters'][0]/2)) + '--'
    logname += 'wdecay_' + str(model_params['weight_decay']) + '--'
    logname += 'elu_' + str(model_params['elu_alpha']) + '--'
    if model_params['regulz_type'] == 'WeightNorm':
        rname = 'WN_'
    if model_params['regulz_type'] == 'BatchNorm':
        rname = 'BN_' + str(model_params['batch_size'])
    if model_params['regulz_type'] == 'InstanceNorm':
        rname = 'IN_'
    if model_params['regulz_type'] == 'LayerNorm':
        rname = 'LN_'
    if model_params['regulz_type'] == 'SpLayerNorm':
        rname = 'SLN_'
    if model_params['regulz_type'] == 'None':
        rname = 'NONE_'
    logname += rname
    return logname