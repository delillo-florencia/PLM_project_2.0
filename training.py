import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import pandas as pd


def setup(rank, world_size):
    print("Setting up...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_teacher_results(result_type: str, batch_number: int, path='/home/developer/PLM_project/data/outputs/'):
    if result_type == 'logi':
        path = path + f'teacher_logi/batch_{batch_number + 1}_logi.pt'
    elif result_type == 'reps':
        path = path + f'teacher_reps/batch_{batch_number + 1}_reps.pt'
    else:
        raise ValueError('Value error: expecting reps or logi string')
    return torch.load(path)

def train(rank, world_size, args):
    setup(0, world_size)
    print("Setup ok")
    # Training parameters
    print("Defining training params...")
    num_epochs = args['num_epochs']
    learning_rate = args['learning_rate']
    weight_rep = args['weight_rep']
    weight_logits = args['weight_logits']
    checkpoints = args['checkpoints']
    cp_dir = args['cp_dir']
    cp_freq = args['cp_freq']
    batch_limit = args['batch_limit']
    run_prefix = args['run_prefix']

    BATCH_SIZE = args['BATCH_SIZE']
    CSV_FILE = args['CSV_FILE']
    REP_LAYER = args['REP_LAYER']
    SEQ_MAX_LEN = args['SEQ_MAX_LEN']
    print("Training params set correctly...")

    # Load dataset
    print("Loading dataset...")

    collection = pd.read_csv(CSV_FILE)
    dataset = HashedProteinDataset(collection)
    sampler = DynamicTaxonIdSampler(num_replicas=world_size,
    rank=rank,
    length_dict=dataset,
    max_batch_size=100,
    max_batch_tokens=100000,
    shuffle=True)

    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: x, shuffle=False)

    print("Data loaded correctly...")

    # Initialize MLflow only on rank 0
    if rank == 0:
        print("Initializing MLflow...")

        mlflow.set_tracking_uri("http://127.0.0.1:8000")
        print("Uri set ok")
        #experiment_id=mlflow.create_experiment("Flor_test")
        #mlflow.set_experiment(experiment_id)
        print("exp set")
    # Load models
    print("Loading models..")
    student_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    teacher_model= FAEsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    teacher_model= FAEsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    teacher_model = teacher_model.to(rank)
    teacher_model = DDP(teacher_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    student_model = student_model.to(rank)
    student_model = DDP(student_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    distillation_loss = DistillationLoss(weight_rep, weight_logits)

    for epoch in range(num_epochs):
        print(f"Running {epoch}..")
        sampler.set_epoch(epoch)  # Ensure randomness across epochs
        cumulative_loss = 0.0
        cum_rep_loss = 0.0
        cum_log_loss = 0.0
        
        for i, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}", unit="batch"):
            if i == batch_limit:
                break
            # extract sequences and names from the batch
            sequences = [item['sequence'] for item in batch]
            names = [item['protein_id'] for item in batch]

            # prepare data for batch conversion
            if names is None:
                names = [f'seq{i}' for i in range(len(sequences))]
            data = list(zip(names, sequences))

            batch_seed = i*BATCH_SIZE

            with mp.Pool() as pool:
                masking = pool.starmap(mask_single, [(n, item, batch_seed) for n, item in enumerate(batch)]) 
            seqs, masked_pos = zip(*masking)

            data_mask = list(zip(names, seqs))

            # check datatype of sequences - str or biotite
            if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
                pass  # all elements are strings
            else:
                data = [(x[0], str(x[1])) for x in data]

            # convert data to batch tensors
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(rank)

            # convert masked data to batch tensors
            masked_batch_labels, masked_batch_strs, masked_batch_tokens = batch_converter(data_mask)
            masked_batch_lens = (masked_batch_tokens != alphabet.padding_idx).sum(1)
            masked_batch_tokens = masked_batch_tokens.to(rank)

            optimizer.zero_grad()

            teacher_model.eval()
                # get results
            with torch.no_grad():
                results = teacher_model(batch_tokens.to(rank), repr_layers=[REP_LAYER], return_contacts=True)

            teacher_reps = get_seq_rep(results, batch_lens, layers=REP_LAYER)
            
            with torch.no_grad():
                results = teacher_model(batch_tokens.to(rank), repr_layers=[REP_LAYER], return_contacts=True)

            res = get_logits(results)
            # trim logits into just masking positions
            masked_logi = []
            for i, positions in enumerate(masked_pos):
                positions = [i+1 for i in positions] #account for <str> token
                masked_logi.append(res[i, positions, :])
            # stack into a tensor with padding (seq have different number of masked pos)
            teacher_logits = pad_sequence(masked_logi, batch_first=True, padding_value=0.0)
            #teacher_logits = load_teacher_results('logi', i).to(rank)
            #teacher_reps = load_teacher_results('reps', i)
            #

            if isinstance(teacher_reps, list):  
                teacher_reps = torch.stack(teacher_reps)  # Convert list to tensor

            teacher_reps = teacher_reps.to(rank)  # Move to device

            student_res = student_model(batch_tokens, repr_layers=[REP_LAYER], return_contacts=False)
            #student_reps = get_seq_rep(student_res, batch_lens, layers=REP_LAYER)
            student_reps = torch.stack(get_seq_rep(student_res, batch_lens, layers=REP_LAYER))  # Convert list to tensor
            student_reps = student_reps.to(rank)  
            
            
            student_masked_res = student_model(masked_batch_tokens, repr_layers=[REP_LAYER], return_contacts=False)
            student_logits = get_logits(student_masked_res)
            

            masked_logi = []
            for i, positions in enumerate(masked_pos):
                positions = [i+1 for i in positions] #account for <str> token
                masked_logi.append(student_logits[i, positions, :])
            # stack into a tensor with padding (seq have different number of masked pos)
            masked_student_logits = pad_sequence(masked_logi, batch_first=True, padding_value=0.0)

            print(student_reps)
            print(f"Student reps: {student_reps.shape}")  # Check shape
            print(f"Student logits: {student_logits.shape}")
            print(f"Teacher reps: {teacher_reps.shape}")  # Check shape
            print(f"Teacher logits: {teacher_logits.shape}")
            loss, rep_loss, log_loss = distillation_loss(teacher_reps, teacher_logits, student_reps, masked_student_logits)
            loss.backward()
            optimizer.step()
    

            cumulative_loss += loss.item()
            cum_rep_loss += rep_loss.item()
            cum_log_loss += log_loss.item()

        avg_loss = cumulative_loss / len(dataloader)
        avg_rep_loss = cum_rep_loss / len(dataloader)
        avg_log_loss = cum_log_loss / len(dataloader)

        if rank == 0:
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)
            mlflow.log_metric("avg_rep_loss", avg_rep_loss, step=epoch)
            mlflow.log_metric("avg_log_loss", avg_log_loss, step=epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

            if checkpoints and (epoch + 1) % cp_freq == 0:
                path = f'{cp_dir}cp_{run_prefix}_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': student_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, path)
                print(f'Checkpoint saved: {path}')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    args = {
        'num_epochs': 5,
        'learning_rate': 1e-4,
        'weight_rep': 0.5,
        'weight_logits': 0.5,
        'checkpoints': True,
        'cp_dir': "/home/developer/PLM_project/data/outputs/checks_flor/",
        'cp_freq': 10,
        'batch_limit': 10000,
        'run_prefix': "10k_rep_log_flor",
        'BATCH_SIZE': 8,
        'CSV_FILE': '/home/developer/PLM_project/data/raw/uniprot_data_500k_sampled_250.csv',
        'REP_LAYER': 6,
        'SEQ_MAX_LEN': 256
    }
    train(0, world_size, args)
    #mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)