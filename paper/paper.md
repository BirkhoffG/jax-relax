---
title: 'ReLax: Efficient and Scalable Recourse Explanation Benchmark using JAX'
tags:
  - Python
  - JAX
  - machine learning
  - interpretability
  - counterfactual explanation
  - recourse
authors:
  - name: Hangzhi Guo
    orcid: 0009-0000-6277-9003
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Xinchang Xiong
    affiliation: 2
  - name: Wenbo Zhang
    affiliation: 1
  - name: Amulya Yadav
    orcid: 0009-0005-4638-9140
    affiliation: 1
affiliations:
 - name: Penn State University, USA
   index: 1
 - name: Duke University, USA
   index: 2
date: 10 December 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Counterfactual explanation[^1] techniques provide contrast cases to individuals adversely affected by ML preictions.
For instance, recourse methods can provide suggestions for loan applicants who have been rejected by a bank's ML algorithm, or give practical advice to teachers handling students at risk of dropping out.
Numerous recourse explanation methods have been recently proposed.
Yet, current research practice focuses on medium-sized datasets (typically around ~50k data points).
This limitation impedes the progress in algorithmic recourse and raises concerns about the scalability of existing approaches. 

[^1]: Counterfactual explanation [@wachter2017counterfactual] and algorithmic recourse [@ustun2019actionable] share close connections [@verma2020counterfactual;@stepin2021survey], which leads us to use these terms interchangeably


To address this challenge, we propose `ReLax`, a JAX-based benchmarking library, designed for efficient and scalable recourse explanations. `ReLax` supports various recourse methods and datasets, demonstrating performance improvements of at least two orders of magnitude over current libraries.
Notably, ReLax can benchmark real-world datasets up to 10 million data points, a 200-fold increase over existing norms, without imposing prohibitive computational costs.



# Statement of need


Recourse and counterfactual explanation methods concentrate on the generation of new instances that lead to contrastive predicted outcomes [@verma2020counterfactual;@karimi2020survey;@stepin2021survey]. Given their ability to provide actionable recourse, these explanations are often favored by human end-users [@binns2018s;@miller2019explanation;@Bhatt20explainable].


Despite progress made in counterfactual explanation research [@wachter2017counterfactual;@mothilal2020explaining;@ustun2019actionable;@upadhyay2021towards;@vo2023feature;@guo2021counternet;@guo2023rocoursenet], current research practices often restrict the evaluation of recourse explanation methods on medium-sized datasets (with under 50k data points). 
This constraint primarily stems from the excessive runtime overhead of recourse generation by the existing open-source recourse libraries [@pawelczyk2021carla;@mothilal2020explaining;@klaise2021alibi].
For instance, as shown in \autoref{fig:speed}, the CARLA library [@pawelczyk2021carla], a popular recourse explanation library, requires roughly 30 minutes to benchmark the adult dataset containing $\sim32,000$ data points. At this speed, it would take CARLA approximately 15 hours to benchmark a dataset with one million samples, and nearly one week to benchmark a dataset with 10 million samples.
Consequently, this severe runtime overhead hinders the large-scale analysis of recourse explanations, impedes the pace of research development of new recourse methods, and raises concerns about the scalability of existing methods being deployed in data-intensive ML applications.

![\label{fig:speed}Runtime comparison of the *adult* dataset between `ReLax` and three open-source recourse librarires (CARLA [@pawelczyk2021carla], DiCE [@mothilal2020explaining], and alibi [@klaise2021alibi].](./figs/speed-compare.pdf)


We present `ReLax` (**Re**course Explanation **L**ibrary using J**ax**), an efficient and scalable benchmarking library for recourse and counterfactual explanations. `ReLax` is the *first* JAX-based library for recourse explanation which leverages language primitives such as vectorization, parallelization, and JIT compilation in JAX [@jax2018github;@frostig2018jax].
`ReLax` is at least two order-of-magnitudes faster than the existing recourse explanation libraries (with a maximum speedup of 404,319.54X, as shown in Figure~\ref{fig:speed}). We further demonstrate that `ReLax` is capable of benchmarking real-world datasets of up to 10M data points with a reasonable amount of computational cost.


In addition, `ReLax` supports a diverse set of recourse methods and datasets. Notably, we implement 9 recourse explanation methods in JAX ranging from non-parametric, semi-parametric, and parametric recourse explanation methods. In addition, we include 14 medium-sized datasets, and one large-scale dataset. Furthermore, we perform comprehensive experiments on both medium-sized and large-sized datasets. 
We have made `ReLax` fully open-sourced. This enables the reproduction of our experiments and supports efficient and scalable benchmarking for newly proposed recourse methods.



# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
