Usage
=====

.. _installation:

Installation
------------

As API User
~~~~~~~~~~~~~~~~~

To use ``nl2prot`` as a tool with its API, first install it with the following steps:

.. code-block:: console

   git clone git@github.com:qzheng75/NL2UniProt.git    # Clone this repository
   cd NL2UniProt                                       # Switch to the working directory
   conda create -n nl2prot python==3.12                # Create a conda env for the project
   conda activate nl2prot                              # Activate conda environment
   bash scripts/installation/install_dependencies.sh   # Install all dependencies

Note that if you may run into a runtime error: ``"FlashAttention only supports Ampere GPUs or newer."``
if your device doesn't have an Ampere GPU (e.g. NVIDIA A40/A100). You can specify a flag ``--no_flash`` as follows to skip installation flash-attn.

.. code-block:: console

   bash scripts/installation/install_dependencies.sh --no_flash

This doesn't affect the API usage, as the current deployed model for the NL2UniProt API doesn't use a model that features Flash Attention.

Since data and trained models need to be downloaded from Google Cloud Storage, you need to contact maintainers of the project for access to data.
After gaining access to Google Cloud Storage, download your credentials and place the .json file under ``google_cloud/``. Then, download the data 
and the trained models by running the following script:

.. code-block:: console

   python scripts/installation/download.py

As Project Developer
~~~~~~~~~~~~~~~~~

If you're a developer of the project (e.g. you want to train your models), you need to perform additional steps to configure wandb for logging and huggingface for downloading pretrained models. Instructions can be found online and won't be specified here.

Use NL2UniProt API
----------------

You can follow `this notebook <https://github.com/qzheng75/NL2UniProt/blob/main/examples/api_use.ipynb>`_ to use the API.

To recommend possibly related sequence ids, first instantiate an ``NL2ProtAPI`` object:

.. code-block:: console

   from nl2prot.api import NL2ProtAPI

   api = NL2ProtAPI()

Then, use the method to get sequence ids from input description(s):

.. autofunction:: api.recommend_sequences(description: str | list[str], top_k: int)
The ``description`` parameter should be either a single description or a list of descriptions,
and use ``top_k`` parameter to control how many sequences you want to return. The sequences are returned
in the order of decreasing similarity to the description.

You can turn output into a pandas DataFrame with two columns: ``accession`` (sequence id in UniProt database) and ``distance``
(the smaller the distance is, the more relevant the sequence is.)

For example:

>>> description1 = (
    "I want a protein with a role in the process "
    + "of adding sugar molecules to other proteins and being involved "
    + "in preventing cell death."
)
>>> description2 = (
    "I want a protein with a specific function in the "
    + "process of creating phospholipids in cell membranes."
)
>>> description3 = "I want a protein with functions related to brain and cell membranes."
>>> out = api.recommend_sequences([description1, description2, description3], 5)
>>> df = pd.DataFrame(out[1])
>>> df.loc[0, 'accession'] # Gives string 'O95674', a UniProt sequence id
>>> df.loc[0, 'distance']  # Gives distance 0.712003

