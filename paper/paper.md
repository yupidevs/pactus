---
title: 'pactus: A Python playground for trajectory classification'
tags:
  - trajectory classification
  - mobility data
  - Python
  - machine learning
authors:
  - name: G. Viera-López
    orcid: 0000-0002-9661-5709
    equal-contrib: true
    corresponding: true 
    affiliation: 1 
  - name: J.J. Morgado-Vega
    orcid: 0000-0001-6067-9172
    equal-contrib: true 
    affiliation: 2
  - name: A. Reyes
    orcid: 0000-0001-7305-4710
    equal-contrib: true
    affiliation: 3
  - name: E. Altshuler
    orcid: 0000-0003-4192-5635
    equal-contrib: false
    affiliation: 3
  - name: Yudivián Almeida-Cruz
    orcid: 0000-0002-2345-1387
    equal-contrib: false
    affiliation: 2
  - name: Giorgio Manganini
    orcid: 0000-0002-5394-4094
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Department of Computer Science, Gran Sasso Science Institute, L'Aquila, Italy
   index: 1
 - name: Department of Artificial Intelligence, University of Havana, Havana, Cuba 
   index: 2
 - name: Group of Complex Systems and Statistical Physics, University of Havana, Havana, Cuba
   index: 3
date: 7 June 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Trajectory classification concerns the forecasting of the class or category of 
a mobile entity based on its observed motion over time. It is a problem that has 
been studied in many different fields, including robotics, behavior analysis, 
mobility pattern mining, and user activity recognition [@da2019survey]. This task
presents multiple challenges for conventional classification models, such as the 
indeterminate length of trajectories [@li2022two], the range of entities that generate 
trajectories [@janczura2020classification; @xiao2017identifying; @bae2022transformer], and the absence of established standards in trajectory datasets 
[@xiao2017identifying; @bae2022transformer]. 

Our study endeavors to lay the foundation for the assessment of innovative 
methods and extend their outcomes to a broader range of datasets. We introduce 
a new framework, referred to as `pactus`, which addresses the challenges of 
trajectory classification by providing direct access to a carefully chosen 
collection of datasets and several trajectory classifiers. `pactus` 
facilitates researchers' ability to experiment with various approaches 
and assess their performance on different types of data.

# Statement of need

The lack of standardization in trajectory datasets, coupled with the difficulty 
of obtaining these datasets for evaluation, poses a significant challenge to 
researchers working in fields related to trajectory classification. Moreover, 
the absence of a reliable and reproducible evaluation methodology makes it 
difficult to compare different methods and assess their performance accurately. 
There is a pressing need for a standardized framework to facilitate the evaluation 
of trajectory classification models in an open and transparent fashion.

In response to these challenges, we have developed `pactus`, a novel framework that 
provides researchers with direct access to a curated list of datasets and relevant 
trajectory classifiers. `pactus` offers a user-friendly interface that encourages 
researchers to include their own datasets or methods on the platform, thereby 
expanding the range of datasets available for evaluation. The library's evaluation 
methodology ensures that results are reproducible and comparable, making it easier 
to identify the most effective trajectory classification methods for specific 
scenarios. Finally, it encourages the production of reproducible research by enabling 
researchers to distribute their findings as simple Python scripts, relying on `pactus` 
for all tasks related to data acquisition, processing, and model evaluation.


# Software Overview

The functionalities of `pactus` can be divided into four different categories as shown in
\autoref{fig:overview}.

![Overview of the resources available in `pactus` coupled with an usage example.\label{fig:overview}](1.pdf)


The selection of datasets was conducted with meticulous care to encompass a broad 
range of trajectories and classification objectives. Our initial selection includes
GeoLife [@zheng2009mining; @zheng2008understanding; @zheng2010geolife], The Starkey 
Project dataset, also known as `Animals' in the	trajectory classification 
community [@rapp2009elk], four different datasets from the the UCI repository 
[@Dua:2019] and two different hurricane datasets, provided by National Hurricane 
Center [@landsea2013atlantic] and the China Meteorological Administration 
[@ying2014overview; @lu2021western] respectively. To ensure consistency, all 
datasets were transformed into a standardized format utilizing the trajectory 
data structures proposed in [@reyes2023yupi]. Datasets are not bundled with the 
software package, but rather will be downloaded and cached automatically upon each
individual access through the library.

In order to mitigate the different-length trajectories on some datasets, `pactus`
is able to extract statistical features from any trajectory and convert an arbitrary
length trajectory into a fixed size feature vector using the procedures described in
[@xiao2017identifying; @zheng2008understanding].

Finally, several classification algorithms can be evaluated on the vectorized
versions of the trajectories (e.g., Random Forest, SVM, KNN) or, alternatively,
classifiers able to handle variable-size inputs (e.g., LSTM or Transformers [@bae2022transformer]) can be evaluated directly on the trajectory data.
In both cases, typical evaluation metrics for classification are computed
automatically for the model being evaluated. 



# References
