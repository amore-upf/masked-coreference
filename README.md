![AMORE-UPF](logos/logo-amore-text-diagonal.png)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      ![UPF](logos/upf-logo2.png)

# Does referent predictability affect the choice of referential form? A computational approach using masked coreference resolution

This repository contains code for the experiments reported in " Does referent predictability affect the choice of referential form? A computational approach using masked coreference resolution " (to appear in Proc. CoNLL 2021). </br>
The masked coreference resolution model used in the paper is built largely based on the coreference resolution model from the paper [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529). Please refer to the [BERT and SpanBERT for Coreference Resolution repository](https://github.com/mandarjoshi90/coref.git) for setup and other information.


## Contents of the repository
Most files are identical to those in the [BERT and SpanBERT for Coreference Resolution repository](https://github.com/mandarjoshi90/coref.git). Some files were modified to enable the fine-tuning on masked coreference resolution, and several files were added to extract context features.
### Modified files
* `evaluate.py`
* `experiments.conf` 
* `independent.py`
* `metrics.py`
* `overlap.py`
* `train.py`

### Added files
* `OntoNotes_ALLENNLP.py`: code used to read and process the English OntoNotes v5.0 data, adapted from the [AllenNLP Models repository](https://github.com/allenai/allennlp-models.git)
* `feature_extraction.py`: functions defined for extracting context features (e.g., frequency, recency, surprisal) 
* `evaluate_and_extract.py`: code used to evaluate the model and to extract features at the same time



## Instructions

### Setup
* Follow the instructions in the [BERT and SpanBERT for Coreference Resolution repository](https://github.com/mandarjoshi90/coref.git) for setup, downloading BERT/SpanBERT models, training, deploying and evaluating models, as well as other information.
* Download the masked coreference resolution model (train_spanbert_base_maskperc15) in [_____](drive link) and put it under the data folder (assumes that $data_dir is set).


### Prediction
Run `GPU=0 python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with an additional key predicted_clusters.


### Evaluation and feature extraction
Feature extraction assumes access to [OntoNotes Release 5.0](https://doi.org/10.35111/xmhb-2b84). Please put CoNLL formatted OntoNotes 5.0 files in the data folder (assumes that $data_dir is set). <br/>

The following line will evaluate the model and produce dataframes of features for analyses.

`GPU=0 python evaluate_and_extract.py <experiment> <phase> <context>`

where 
* `<experiment>` specifies the experiment that you would like to use, e.g.,: spanbert_base, train_spanbert_base_maskperc15. Experiment configurations are found in `experiment.conf`.
* `<phase>` can be `eval` or `test`. `<phase>` is set to be `eval` by default and the model is evaluated on the dev set. To run the model on the test set, you need to specify this parameter to be `test`.
* `<context>` specifies the scope of context you would like the model to use for coreference resolution. The model uses both sides of the context if not specified, and only uses left context if this is set to be `True`.



## Citation
The system is described in this paper: Does referent predictability affect the choice of referential form? A computational approach using masked coreference resolution (arxiv link)

```
@inproceedings{ aina2021referent,
    title     = {Does referent predictability affect the choice of referential form? A computational approach using masked coreference resolution},
    author    = {Aina, Laura and Liao, Xixian and Boleda, Gemma and Westera, Matthijs},
    booktitle = {Proceedings of the 25th Conference on Computational Natural Language Learning},
    pages     = {(to appear)},
    month     = {November},
    year      = {2021},
    address   = {Online},
    publisher = {Association for Computational Linguistics},
}
```

Additionally, if you use the pretrained *BERT*-based coreference model (or this implementation), please cite the paper, [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091).
```
@inproceedings{joshi2019coref,
    title={{BERT} for Coreference Resolution: Baselines and Analysis},
    author={Mandar Joshi and Omer Levy and Daniel S. Weld and Luke Zettlemoyer},
    year={2019},
    booktitle={Empirical Methods in Natural Language Processing (EMNLP)}
}
```


If you use the pretrained *SpanBERT*-based coreference model (or this implementation), please cite the paper, [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).
```
@article{joshi2019spanbert,
    title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
    author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
    year={2019},
    journal={arXiv preprint arXiv:1907.10529}
}
```

## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No. 715154). Matthijs Westera also received funding from Leiden University (LUCL, SAILS). We are grateful to the NVIDIA Corporation for the donation of GPUs used for this research. This paper reflects the authors' view only, and the EU is not responsible for any use that may be made of the information it contains.


![(ERC logo)](logos/LOGO-ERC.jpg)      &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;       ![(EU flag)](logos/flag_yellow_low.jpeg)
