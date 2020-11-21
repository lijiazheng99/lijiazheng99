# Papers I read
Listed papers are what I have read and what I find useful to my researches. I can't guarantee correctness of my short notes for each paper. But if you find me misunderstand some of these papers, welcome to contact me and share your thoughts :)       

I'm open for research collaborations on natural language processing, information retrieval and machine learning.

Topics:    
[General NLP](#General-NLP) | [Explainable Model](#explainable-model-in-nlp-post-hoc) | [Adversarial](#Adversarial-Training) | [Style Transfer](#Style-Transfer) | [Counterfactual](#Counterfactual-Generation) |  [Evaluation](#Evaluation) | [Fintech](#Financial-Application)

Related links:    
[HuggingFace's Transformer](https://huggingface.co/) | [Weights & Biases](https://wandb.ai/) | [Allen NLP](https://docs.allennlp.org/)              

[Google research's github](https://github.com/google-research) | [Facebook research's github](https://github.com/facebookresearch)        

Big groups:        
[UCL NLP](https://nlp.cs.ucl.ac.uk/) | [UCL DARK](https://ucl-dark.github.io/) | [Edinburgh NLP](https://edinburghnlp.inf.ed.ac.uk/) | [Cambridge NLP](https://www.cl.cam.ac.uk/research/nl/publications.html)

## General NLP

[**(2020) Experience Grounds Language**](https://arxiv.org/abs/2004.10151)      
A general review of how NLP will develop in future.  

[**(2020) HuggingFace's Transformers: State-of-the-art Natural Language Processing**](https://arxiv.org/abs/1910.03771)-[HuggingFace's Transformer](https://huggingface.co/)     
NLPer's old friend   

[**(2018) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805)      
BERT          
[A towards data science artical on explain what is bert](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)     

[**(2017) Attention Is All You Need**](https://arxiv.org/abs/1706.03762)           
Transformer

## Explainable Model in NLP (Post-hoc)

[**(2017) Understanding Neural Networks through Representation Erasure**](https://arxiv.org/abs/1612.08220)      
Explain nerual network with word importance.

[**(2018) Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs**](https://arxiv.org/abs/1801.05453)-[Code](https://github.com/jamie-murdoch/ContextualDecomposition)    
Contextual Decomposition algorithm.

[**(2019) Interpretable machine learning: definitions, methods, and applications**](https://arxiv.org/abs/1901.04592)     
A framework for interpretability.

[**(2019) Hierarchical interpretations for neural network predictions**](https://arxiv.org/abs/1806.05337)-[Code](https://github.com/csinva/hierarchical-dnn-interpretations)      
Update on CD algorithm, purposed ACD, can interpret phrase level.

[**(2019) Disentangled Attribution Curves for Interpreting Random Forests and Boosted Trees**](https://arxiv.org/abs/1905.07631)-[Code](https://github.com/csinva/disentangled-attribution-curves)      
Disentangled Attribution Curves (DAC), a method to provide interpretations of tree ensemble methods. 

[**(2019) Interpretations are useful: penalizing explanations to align neural networks with prior knowledge**](https://arxiv.org/abs/1909.13584)-[Code](https://github.com/laura-rieger/deep-explanation-penalization)      
contextual decomposition explanation penalization, hummm like adversarial attack?

[**(2020) Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models**](https://arxiv.org/abs/1911.06194)-[Code](https://github.com/INK-USC/hierarchical-explanation-neural-sequence-models)      
Update on ACD, proposed SCD and SOC.

## Adversarial Training

[**(2018) Towards Deep Learning Models Resistant to Adversarial Attacks**](https://arxiv.org/abs/1706.06083)     
Adversarial attack algorithm FGSM and PGD.

[**(2020) Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**](https://arxiv.org/abs/1907.11932)     
Adversarial training by positive <-> negative.

## Style Transfer

[**(2019) Mask and Infill: Applying Masked Language Model to Sentiment Transfer**](https://arxiv.org/abs/1908.08039)      
Use Mask and Infill replace sentiment words to do style transfer.

[**(2019) A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer**](https://arxiv.org/abs/1905.10060)-[Code](https://github.com/luofuli/DualRL)     
sentiment: positive <-> negative, content: formal <-> informal

## Counterfactual Generation

[**(2020) Learning the Difference that Makes a Difference with Counterfactually-Augmented Data**](https://arxiv.org/abs/1909.12434)-[Data](https://github.com/acmi-lab/counterfactually-augmented-data)     
Same as what title tells

## Evaluation

[**(2018) Stress Test Evaluation for Natural Language Inference**](https://arxiv.org/abs/1806.00692)    
.

[**(2020) Beyond Accuracy: Behavioral Testing of NLP models with CheckList**](https://arxiv.org/abs/2005.04118)-[Code](https://github.com/marcotcr/checklist)    
Good accuracy on score doesn't means model robustness

## Financial Application

[**(2019) What You Say and How You Say It Matters: Predicting Financial Risk Using Verbal and Vocal Cues**](https://www.aclweb.org/anthology/P19-1038)

## Other
