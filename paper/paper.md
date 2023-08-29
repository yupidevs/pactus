---
title: 'pactus: A Python framework for trajectory classification'
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
  - name: E.~Altshuler
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
trajectories [@janczura2020classification; @xiao2017identifying; @bae2022transformer], 
and the absence of established standards in trajectory datasets 
[@xiao2017identifying; @bae2022transformer]. 

Our study endeavors to lay the foundation for the assessment of innovative 
methods and extend their outcomes to a broader range of datasets. We introduce 
a new framework, referred to as `pactus`, which addresses the challenges of 
trajectory classification by providing direct access to a carefully chosen 
collection of datasets and several trajectory classifiers. `pactus` 
facilitates researchers' ability to experiment with various approaches 
and assess their performance on different types of data. A comprehensive software 
documentation is provided on 
([https://pactus.readthedocs.io/en/latest/](pactus.readthedocs.io)).

# Statement of need

In recent years, several software libraries have emerged, aiming to automate trajectory 
data analysis. Within the R community, there are various available tools 
[@joo2020navigating]. Recognizing the popularity and extensive usage of Python, the `traja`
software [@shenk2021traja] was developed to integrate different analysis techniques for 
two-dimensional trajectories, primarily focusing on animal behavioral analysis. Additionally, 
the `yupi` library [@reyes2023yupi] was created to handle trajectory analysis for applications 
involving an arbitrary number of dimensions.

Although these libraries offer valuable tools for trajectory classification, such as 
classification models and feature extraction from trajectories, they were not specifically 
designed for this task. Consequently, contemporary research on trajectory classification
faces limitations in terms of evaluation, often considering only a limited number of datasets 
or reporting only a reduced set of metrics [@bae2022transformer].

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


# Pactus Software Library

The functionalities of `pactus` can be divided into modules: Data handling, Feature extraction,
Classification models and Evaluation.

## Data handling

The library provides direct access to some of the most commonly used datasets for trajectory 
classification. The selection of datasets was conducted with meticulous care to encompass a broad 
range of trajectories and classification objectives. Our initial selection includes
GeoLife [@zheng2009mining; @zheng2008understanding; @zheng2010geolife], The Starkey 
Project dataset, also known as `Animals` in the	trajectory classification 
community [@rapp2009elk], four different datasets from the the UCI repository 
[@Dua:2019] and two different hurricane datasets, provided by National Hurricane 
Center [@landsea2013atlantic] and the China Meteorological Administration 
[@ying2014overview; @lu2021western] respectively. To ensure consistency, all 
datasets were transformed into a standardized format utilizing the trajectory 
data structures proposed in [@reyes2023yupi]. Datasets are not bundled with the 
software package, but rather will be downloaded and cached automatically upon each
individual access through the library. A complete guide on how to use custom datasets or
requesting the inclusion of new datasets into `pactus` can be found in the documentation.

## Feature extraction

In order to mitigate the different-length trajectories on some datasets, `pactus`
is able to extract statistical features from any trajectory and convert an arbitrary
length trajectory into a fixed size vector whose components are engineered features 
typically used in the literature [@xiao2017identifying; @zheng2008understanding].

Users can implement their own method to perform this conversion, and an example on how
to do it can be found in the documentation. However, there is a default method that uses
all the features computed by the `yupi` library.

## Classification models and Evaluation

Several classification algorithms are included in  `pactus`. Some of them can be evaluated 
on the vectorized versions of the trajectories (e.g., Random Forest, SVM, KNN). In other cases
the classifiers are able to handle variable-size inputs (e.g., LSTM or Transformers 
[@bae2022transformer]) and can be evaluated directly on the trajectory data. In both cases, 
typical evaluation metrics for classification are computed automatically for the model being evaluated. 

## Overview

All the functionalities of the library can be integrated in a single script. \autoref{fig:overview}
shows an example on how to use `pactus` for training and evaluating a Random Forest model using the
Starkey Project dataset, also known as `Animals`.

![Overview of the resources available in `pactus` coupled with an usage example.\label{fig:overview}](1.pdf)


# Conclusions

The software presented with this work, `pactus`, addresses typical challenges faced in trajectory 
classification research. By providing researchers with direct access to curated datasets and trajectory 
classifiers, `pactus` enhances the availability of resources for evaluation. It is concieved with extensibility 
in mind, encouraging researchers to contribute their own datasets and methods. The evaluation methodology ensures 
reproducibility and comparability of results, facilitating the identification of effective trajectory classification 
methods for specific scenarios. Additionally, pactus promotes reproducible research by enabling researchers to 
distribute their findings as Python scripts, relying on pactus for data acquisition, processing, and model 
evaluation. Overall, pactus offers a valuable tool for researchers in the field of trajectory classification, 
addressing key challenges and facilitating future advancements in the field.

# References
