NL2UniProt
==========

Getting Started
---------------

To get started with the NL2UniProt project, follow these steps:

1. **Create a Conda Environment**

    Open a terminal and run the following command to create a new Conda environment:

    ```
    conda create --name nl2uniprot python=3.12
    ```

2. **Activate the Conda Environment**

    ```
    conda activate nl2uniprot
    ```

3. **Install Dependencies**

    Run the installation script to install all necessary dependencies. Note that this step must be performed on a GPU device,
    as the installation script will install the GPU version of PyTorch. If you don't want to install Flash Attention version of ESM,
    add a flag `--no_flash` to the installation script.
    
    With Flash Attention:

    ```
    bash scripts/installation/install_dependencies.sh
    ```

    or without Flash Attention:

    ```
    bash scripts/installation/install_dependencies.sh --no_flash
    ```

4. **Setup utilities for the project**

    This project currently supports wandb logging. To use wandb logging, you need to setup wandb by running the following command.
    Make sure that you have a wandb account before running this command.

    ```
    wandb login
    ```

    This project interacts with Google Cloud Services. To use Google Cloud Services, you need to setup Google Cloud SDK and acquire proper
    credentials. Contact the project maintainers for obtaining necessary credentials.

5. **Download data and trained models for the project**

    Run the following command to download the necessary data and trained models from GCS for the project:

    ```
    python scripts/installation/download.py
    ```

You are now ready to start using the NL2UniProt project!

Refer to `docs/contribute_guide.md` for contributing to this project.
