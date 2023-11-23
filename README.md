# KTH_DeepLearning

## Erkenntnisse / Ideen für Experimente
- Bei Formel Seite 23 (paper) ist min und max vertauscht gegenüber t_steps im code, max - min macht für mich mehr Sinn!
- 

## Fragen an Betreuer
- From where should we use the dataset? Official paper one or external CIFAR-10?
- For what does t_i hat stand for? And why not use t_i?

- Antwort: Code kann man vom Repo nehmen, aber eigene Ideen umsetzten, welche nicht im paper sind! Und das begründen, was man macht! Hyperparameter search über etwas was sie noch nicht gemacht haben! Neue Methode, Experimente!


## Setup guide
- Run environment.yml to create a conda environment with most necessary packages
- 'pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl'
- Download folder 'torch_utils'
- Download folder 'externals' for pretrained classifier model 'adm_model_definition.py'
- Download dataset 'true_data.npz' from https://drive.google.com/drive/folders/15sslg1DLiFznioOOj0RjqWTAUpQUf8bT?usp=sharing
- Download EDM diffusion model 'edm-cifar10-32x32-uncond-vp.pkl' from https://drive.google.com/drive/folders/15sslg1DLiFznioOOj0RjqWTAUpQUf8bT?usp=sharing
- Download pretrained ADM classifier model '32x32_classifier.pt' from  https://drive.google.com/drive/folders/15sslg1DLiFznioOOj0RjqWTAUpQUf8bT?usp=sharing
- Download pretrained discriminator model 'discriminator_60_uncond_pretrained.pt' from https://drive.google.com/drive/folders/15sslg1DLiFznioOOj0RjqWTAUpQUf8bT?usp=sharing


## Information, grading from KTH
### Detailed Assessment Criteria

- [ ] Satisfactory Project (E-D): The general criteria for a "Satisfactory Project" should be clear, i.e. successful reimplementation of the paper's approach, replicating some basic main experiments of the paper, and producing a corresponding report of acceptable quality.
- [ ] Good-Very Good Project(D-C-B): As is discussed in one of the lectures of the course, there are several ways that a paper can be examined from the perspective of best practices in machine learning research, for instance by checking the proper evaluation procedure (train, validation, test splits), sensitivity to hyper-parameters, sensitivity to the design choices, statistical significance of the reported results, overfitting to the test set, the proper and complete choice of relevant baselines and very importantly ablation studies of different components of the approach.
- [ ] Excellent Project (B-A): Going beyond the scope of the papers method and experiments can be done in several ways that the students can propose. It's also perfectly fine if the extension(s) are inspired by similar extensions present in other papers from the literature, of course proper reference should be provided. Some examples are:

    - [ ] fair, conclusive but new comparisons with few other papers, particularly listed ones under the same topic, using their available online codes (strong preference for the official code)
    - [ ] novel, interesting, and original application to a different task
    - [ ] novel, interesting, and informative new metrics and/or ways of evaluating the methods
    - [ ] novel, interesting, and/or informative modification or incremental improvement of the proposed method
    - [ ] justifiable combination of the methods proposed in two or more papers and corresponding experiments
 

### Bonus points
Added to the general guidelines, a project can stand out of the submitted projects for one reason or another the following bonus points are some examples that can boost your final grade:

- [ ] Complexity of the new application / dataset
- [ ] Difficulty of the implementation: we understand that some papers might be significantly more difficult than others to implement.
- [ ] Successful reimplementation in a deep learning framework for which an online public repository is not available
- [ ] Noticeably interesting and informative observations from the experiments (both positive and negative
- [ ] A comparative computational, memory, and storage complexity analysis of your project paper with other papers.
- [ ] A very well-written report: this will be judged by the readability of the report, its organization, use of original figures, innovative representations such as notations, summary tables, etc.
- [ ] Deep, high-quality, and thorough writing of the broader impact
- [ ] Thorough summary of related works, especially the works published in last instance of AI conferences and journals including but not limited to: NeurIPS 2021-23, ICML 2022-23, ICLR 2022-23, AAAI 2022-23, UAI 2022-23, AIStats 2022-23, ECCV 2022, ICCV 2023, CVPR 2022-23, EMNLP 2022-23, ACL 2022-23, ICASSP 2022-23, and their corresponding workshops.


### Penalization points
Certain points can push down the project grade from the provided general guidelines, for instance:
- Limited discussions, merely reporting results without further analysis
- Hardly-readable report
- Incorrect presentation or description of the paper's method and/or background
- A poorly provided peer-review 

### Some final words
Some final words about doing and grading a project. We should mention here that grading a project-based course is generally a complicated process  and that we have not (or in general there is no clear consensus) yet figured out an ideal way of doing it. We believe it is additionally challenging for this course that deals with state-of-the-art papers on various cutting-edge research topics. Still we didn't opt for a written exam that will be more well-defined and causes less headache for both students and teachers since we think it's more relevant for the intended learning outcomes of such an advanced course, but also it's just more fun to do a project! 
Every year, based on the feedback we receive from the students and our own experiences, we try to change and hopefully concretize the grading criteria. As such the final grading might inevitably appear to some students as slightly arbitrary. So, keep that in mind when reading the criteria and guideline and take notes of what you think we can improve for this and then put them when the course evaluation forms are sent to you.
