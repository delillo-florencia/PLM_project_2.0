import torch

def compare_checkpoints(ckpt1_path, ckpt2_path):
    # Load checkpoints (map to CPU to avoid device issues)
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")
    
    # Extract state dictionaries
    state_dict1 = ckpt1['state_dict']
    state_dict2 = ckpt2['state_dict']
    
    all_equal = True

    # Check that both state dictionaries have the same keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    if keys1 != keys2:
        print("The checkpoints have different keys!")
        missing_in_ckpt2 = keys1 - keys2
        missing_in_ckpt1 = keys2 - keys1
        if missing_in_ckpt2:
            print("Keys missing in second checkpoint:", missing_in_ckpt2)
        if missing_in_ckpt1:
            print("Keys missing in first checkpoint:", missing_in_ckpt1)
        all_equal = False

    # Compare each parameter tensor
    for key in state_dict1.keys():
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                print(f"Parameter '{key}' differs between checkpoints.")
                all_equal = False

    if all_equal:
        print("All weights are equal between the checkpoints.")
    else:
        print("Some weights differ between the checkpoints.")

if __name__ == '__main__':
    # Replace these paths with the actual checkpoint file paths
    ckpt1_path = "/home/developer/Projects/PLM_project_2.0/src/data/outputs/outputs/checkpoint_step_step=200.ckpt"
    ckpt2_path = "/home/developer/Projects/PLM_project_2.0/src/data/outputs/outputs/checkpoint_step_step=400.ckpt"
    
    compare_checkpoints(ckpt1_path, ckpt2_path)
