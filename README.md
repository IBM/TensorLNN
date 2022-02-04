# Scaling of training and inference of Propositional Logical Neural Networks with applications to Text Word Common Sense Games and Wordnet Sense Disambiguation. 

The repository contains the code base for TensorLNN, a scalable implementation of the training and inference of Propositional Logical Neural Networks using Lukasevic logic. 


## Setting up the environment 

```bash
conda create -n tensorlnn python=3.9 numpy ipython matplotlib tqdm scipy
conda activate tensorlnn
conda install pytorch=1.10.0 torchvision torchaudio -c pytorch
```
## LOA benchmark using Tensor Logical Neural Networks.

The loa benchmark employs a supervised training of TensorLNN where the training samples and ground truths are generated and the underlying formula is to be determined by training through the TensorLNN. 

The benchmark  can be run by simply executing the following commands:

```bash
cd examples/loa
python atloc_lnn.py
```

The top level program is `atloc_lnn.py`. The tensorLNN model is defined using the statement  `tensorlnn.NeuralNet(num_inputs, gpu_device, nepochs, lr, optimizer)`. Here,

```json
{
        "num_inputs" : "number/of/nodes/whose/AND/is/to/be/determined",
        "gpu_device": /set/to/true/or/false/depending/on/gpu/or/cpu/run,
        "nepochs" : "number/of/epochs/to/train",
        "lr" : "learning/rate",
        "optimizer" : "SGD/or/AdamW",
}
```

A simpler program to test the loa based on some random sample generation is implemented in `basic_lnn.py`. The program can be run by executing the following commands:

```bash
cd examples/loa
python basic_lnn.py <num_samples> <num_inputs>
```

This will generate num_samples random positive training samples, each being a vector of length num_inputs, and accordingly train the propositional LNN on those samples.


## Scaling of Logical Neural Networks for Word Sense Disambiguation (WSD)

TensorLNN for WSD involves unsupervised training using initial bounds of the nodes, and senses defined by universes.
This is done by executing the following commands.


```bash
cd examples/wsd
python wsd_main.py
```

The top level program is `wsd_main.py`. The model construction and training parameter inputs need to be specified in `config.json`. A typical input is of the form:

```json
{
        "univ_dir" : "path/to/universe/data/folder",
        "group_size" : "number/of/universes/merged/in/megauniverse",
        "nepochs" : "number/of/epochs/to/train",
        "lr" : "learning/rate",
        "inf_steps" : "number/of/inference/steps/in/each/epoch",
        "fwd_method" : "baseline/or/checkpoint",
        "checkpoint_phases" : "number/of/inference/steps/between/checkpoints",
        "smooth_alpha" : "exponentiation/parameter/for/smooth/aggregation",
        "clamp_thr_lb" : "lower/theshold/for/claming/weights/and/bias",
        "clamp_thr_ub" : "upper/theshold/for/claming/weights/and/bias",
        "eps" : "some/low/positive/epsilon",
        "optimizer" : "AdamW",
        "gap_slope" : \gamma,
        "contra_slope" : \zeta,
        "vacuity_mult" : \nu,
        "logical_mult" : \lambda,
        "bound_mult" : \beta
}
```

Note: 
--- "univ_dir" should have "global" and "local" subdirectories and "universes.txt". "universes.txt" should contain list of the universe ids. The "global" subdirectory should contain npz file specifying the adjacency matrix of the global LNN. The "local" subdirectory should have for each of the universe ids one subdirectory of the same name and that shall contain for that universe (i) npz file for the adjacency matrix of AND net (ii) npz file for the adjacency matrix of NOT net (iii) bounds file.  

---"checkpoint_phases" is used only when "fwd_method" is set as "checkpoint". It should be set to a number that divides "inf_steps".

---Given bounds (<img src="https://render.githubusercontent.com/render/math?math=L,U">) on each node <img src="https://render.githubusercontent.com/render/math?math=v"> and weights <img src="https://render.githubusercontent.com/render/math?math=w"> and bias <img src="https://render.githubusercontent.com/render/math?math=b">,  the total loss after inference is given by: 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Loss =  \beta \cdot ( \gamma^2 \cdot GL %2B \zeta^2  \cdot CL)  %2B \lambda \cdot LL %2B \nu \cdot VL ">
</p>



where Gap Loss (<img src="https://render.githubusercontent.com/render/math?math=GL">),  Contradiction Loss (<img src="https://render.githubusercontent.com/render/math?math=CL">), Logical Loss (<img src="https://render.githubusercontent.com/render/math?math=LL">) and Vacuity Loss (<img src="https://render.githubusercontent.com/render/math?math=VL">) are respectively given as:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=CL = \sum_{v} [ relu(L(v)-U(v)) ]^2 ">
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=GL = \sum_{v} [ relu(U(v)-L(v)) ]^2 ">
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=LL = \sum_{v} \left [(\sum_{u\in in(v)}{relu(b(v)-w(u,v)})/|in(v)|\right] ">
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=VL = \sum_{v} \left [1-b(v)\right]^2 ">
</p>


