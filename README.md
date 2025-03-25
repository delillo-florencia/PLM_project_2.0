# PLM_project_2.0

# **Small Lessons, Big Learner:** Fine Tuning of ESM-2 Protein Language Model Through Knowledge Distillation
> *Authors:* Edir Sebastian Vidal Castro, Florencia De Lillo, Kacper Maciejewski, Rodrigo Gallegos Dextre <br />
> *Supervisor:* Jonathan Funk <br />

## Repository outline
TBA
```
PLM_PROJECT_2.0/
├── src/
│   ├── configs/                    # Config handling modules
│   │   ├── __init__.py
│   ├── data/                    # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py           # HashedProteinDataset class
│   │   ├── sampler.py           # DynamicTaxonIdSampler class
│   │   └── sequence.py          # Sequence class
│   ├── lightning/                  # Lightning-specific modules
│   │   ├── __init__.py
│   │   ├── callbacks.py            # For updating sampler epoch
│   │   ├── data_module.py          # ProteinDataModule
│   │   └── pyl_module.py         # ProteinReprModule
│   │
│   ├── models/                  # Model-related code
│   │   ├── __init__.py
│   │   └── model_selector.py    # ModelSelector class
│   │
│   ├── tests/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── check_checkpoints.py    
│   │   ├── get_reps_test.py   
│   │   ├── get_reps_token_size.py          
│   │   └── test_faem.py       
│   │
│   ├── tools/                  # Tools-related code
│   │   ├── __init__.py
│   │   ├── download_uniref100.sh          
│   │   └── run_hashing.py 
│   │
│   ├── training/                   # Training module
│   │   ├── __init__.py
│   │   ├── get_reps.py    
│   │   ├── pyl_training.py   
│   │   └── training.py       
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py           # get_latest_version function
│   │   ├── loss_functions.py    # loss
│   │   └── model_utils.py       # get_seq_rep, get_logits functions
│   │   └── token_mask.py        # masking
│   │
│   └── __init__.py              # Package initialization
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # Project documentation
```

