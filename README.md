# PLM_project_2.0

# **Small Lessons, Big Learner:** Fine Tuning of ESM-2 Protein Language Model Through Knowledge Distillation
> *Authors:* Edir Sebastian Vidal Castro, Florencia De Lillo, Kacper Maciejewski, Rodrigo Gallegos Dextre <br />
> *Supervisor:* Jonathan Funk <br />

Repository for the final project of the 02456 Deep Learning (Fall 2024) course at Technical University of Denmark.

## Repository outline
```
PLM_PROJECT_2.0/
│
|
├── src/                                        # Main source code
|
│   │
│   ├── training/                               # Training loop for knowledge distilation
│   │   ├── get_logits.py                       # Precomputes logits from the student model
│   │   ├── get_reps.py                         # Precomputes representations from the student model
│   │   └── training_loop.py                    # Knowledge-destillation loop on precomputed results
│   │
│   ├── utils/                                  # Training-related code
│   │   ├── data_utils.py                       # Data loader with taxonomy-oriented batching
|   |   ├── loss_functions.py                   # Dual-loss implementation
│   │   └── token_mask.py                       # Multiprocessing-enabled sequence masking
```

