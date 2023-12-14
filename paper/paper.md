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
For instance, as shown in \autoref{fig:speed}, the CARLA library [@pawelczyk2021carla] requires roughly 30 minutes to benchmark the adult dataset containing $\sim32,000$ data points. At this speed, it would take CARLA approximately 15 hours to benchmark a dataset with 1 million samples, and nearly one week to benchmark a 10-million dataset.
Consequently, this severe runtime overhead hinders the large-scale analysis of recourse explanations and the research development of new recourse methods.

![\label{fig:speed}Runtime comparison of the *adult* dataset between `ReLax` and three open-source recourse librarires (CARLA [@pawelczyk2021carla], DiCE [@mothilal2020explaining], and alibi [@klaise2021alibi].](./figs/speed-compare.pdf)


In this work, we present `ReLax` (**Re**course Explanation **L**ibrary using J**ax**), the *first* recourse explanation library in JAX [@jax2018github;@frostig2018jax]. Our contributions are three-fold:

* (Fast and Scalable System) `ReLax` is an *efficient and scalable benchmarking library* for recourse and counterfactual explanations.
* (Comprehensive set of Methods) `ReLax` implements 9 recourse explanation methods. In addition, `ReLax` include 14 medium-sized datasets, and one large-scale dataset.
* (Extensive Experiments) We perform comprehensive experiments on both medium-sized and large-sized datasets, which showcases the usability and scalability of the library.


## Efficiency and Scalability in `ReLax`


`ReLax` supports three recourse generation strategies: *sequential*, *vectorized*, and *parallelized* strategy. In particular, the *sequential* generation strategy is inefficient, albeit being adopted in most existing libraries. On the other hand, the *vectorized* and *parallelized* strategies play a vital role in equipping `ReLax` to benchmark large-scale datasets with a practical computational cost. In addition to these, `ReLax` further enhances its performance by fusing inner recourse generation steps via the Just-In-Time (JIT) compilation. Together, `ReLax` ensures efficient and scalable performance across diverse data scales and complexities.


## Recourse Methods & Datasets


`ReLax` implements nine recourse methods using JAX including (i) three non-parametric methods (VanillaCF [@wachter2017counterfactual], DiverseCF [@mothilal2020explaining], GrowingSphere [@laugel2017inverse]); (ii) three semi-parametric methods (ProtoCF [@van2019interpretable], C-CHVAE [@pawelczyk2020learning], CLUE [@antoran2021clue]); and (iii) three parametric methods (VAE-CF [@mahajan2019preserving], CounterNet [@guo2021counternet], L2C [@vo2023feature]).


Furthermore, we gather 14 medium-sized binary-classification tabular datasets. We also benchmark over the forktable dataset [@ding2021retiring] for predicting individuals' annual income. This US censuring dataset contains $\sim 10$ million data points.
To our knowledge, this is the first attempt to benchmark a dataset at the scale of 10 million data points in the recourse explanation community.


![\label{fig:comparison}Comparison of recourse method performance across 14 medium-sized datasets. It is desirable to achieve *high* validity, *low* proximity, and *low* runtime.](./figs/results.pdf)

![\label{fig:strategy_runtime}Runtime comparison of different recourse generation strategies on the forktable dataset.](./figs/strategy_compare.pdf)

## Experimental Results

\autoref{fig:comparison} compares the validity, proximity, and runtime achieved by nine recourse methods averaged on 14 medium-sized datasets. In particular, validity and proximity measure the quality of the generated counterfactual explanations. We observe that CounterNet and Growing Sphere achieve the best validity score, and C-CHVAE achieves the best proximity score. 
In terms of runtime, all recourse methods complete the entire recourse generation process within 10 seconds, while CounterNet and VAECF outperform others by completing under 2 seconds.  


\autoref{fig:strategy_runtime} compares the runtime for each recourse explanation method in adopting the vectorized and parallelized strategies on the forktable dataset (with 10M data points). First, `ReLax` is highly efficient in benchmarking the large-scale dataset, with the maximum runtime being under 30 minutes. 
On the other hand, by estimation, existing libraries should take at least one week to complete recourse generation on datasets at this scale.
In addition, the parallelized strategy cuts the runtime by roughly 4X, which demonstrates that `ReLax`'s potential in benchmarking even larger datasets.

# References
