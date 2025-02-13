# PLM_project_2.0

# **Small Lessons, Big Learner:** Fine Tuning of ESM-2 Protein Language Model Through Knowledge Distillation
> *Authors:* Edir Sebastian Vidal Castro, Florencia De Lillo, Kacper Maciejewski, Rodrigo Gallegos Dextre <br />
> *Supervisor:* Jonathan Funk <br />

Repository for the final project of the 02456 Deep Learning (Fall 2024) course at Technical University of Denmark.

## Repository outline
```
PLM_PROJECT/
│
├── bin/                                        # Playground directory for tries and errors
│   ├── example_data/                           # Examples of input data format
│   │   ├── uniprot_data.cvs
│   │   └── uniref_id_UniRef100_A0A003_OR_id_UniR_2024_11_17.ts
|   |
│   ├── data_preprocessing_prototype.ipynb      # Legacy code for data loading
│   ├── testing_loss_functions.ipynb            # Legacy code for loss implementation
│   └── testing_training_loop.ipynb             # Legacy code for trainng loop
|
├── src/                                        # Main source code
│   ├── evaluation/                             # Create plots from training's output
│   │   ├── acc_perplex.py                      # Create boxplots for the poster
|   |   ├── download_mlflow_results.py          # Get tracked training parameters and metrics
│   │   └── line_plots.py                       # Create line plots for the poster
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

## Poster
![Project description](bin/img/poster.png)