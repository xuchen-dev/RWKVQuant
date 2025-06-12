from datasets import load_dataset


def get_calib_dataset(numbers=16):
    dataset_load = []

    ds_hellaswag = load_dataset("Rowan/hellaswag")
    ds_hellaswag = ds_hellaswag['test'] # ds['ctx_a']

    for i, data in enumerate(ds_hellaswag):
        if i==numbers:
            break
        dataset_load.append( data['ctx_a'] )


    ds_piqa = load_dataset("ybisk/piqa")
    ds_piqa = ds_piqa['test']['goal'] # 

    for i, data in enumerate(ds_piqa):
        if i==numbers:
            break
        dataset_load.append( data )

    ds_ai2_arc = load_dataset("allenai/ai2_arc", "ARC-Easy")
    ds_ai2_arc = ds_ai2_arc['test']['question'] # 

    for i, data in enumerate(ds_ai2_arc):
        if i==numbers:
            break
        dataset_load.append( data )

    ds_ai2_arc_c = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    ds_ai2_arc_c = ds_ai2_arc_c['test']['question'] # 

    for i, data in enumerate(ds_ai2_arc_c):
        if i==numbers:
            break
        dataset_load.append( data )

    ds_sciq = load_dataset("allenai/sciq")
    ds_sciq = ds_sciq['test']['question'] 

    for i, data in enumerate(ds_sciq):
        if i==numbers:
            break
        dataset_load.append( data )

    ds_w = load_dataset("allenai/winogrande", 'winogrande_xs', trust_remote_code=True)
    ds_w = ds_w['test']['sentence'] 

    for i, data in enumerate(ds_w):
        if i==numbers:
            break
        dataset_load.append( data )

    return dataset_load 