# Improving Health Mention Classification Through Emphasising Literal Meanings

Dataset for our 2023 The Web Conference paper [Improving Health Mention Classification Through Emphasising Literal Meanings: A Study Towards Diversity and Generalisation for Public Health Surveillance ](https://dl.acm.org/doi/10.1145/3543507.3583877)

## Abstract
People often use disease or symptom terms on social media and online forums in ways other than to describe their health. Thus the NLP health mention classification (HMC) task aims to identify posts where users are discussing health conditions literally, not figuratively. Existing computational research typically only studies health mentions within well-represented groups in developed nations. Developing countries with limited health surveillance abilities fail to benefit from such data to manage public health crises. To advance the HMC research and benefit more diverse populations, we present the Nairaland health mention dataset (NHMD), a new dataset collected from a dedicated web forum for Nigerians. NHMD consists of 7,763 manually labelled posts extracted based on four prevalent diseases (HIV/AIDS, Malaria, Stroke and Tuberculosis) in Nigeria. With NHMD, we conduct extensive experiments using current state-of-the-art models for HMC and identify that, compared to existing public datasets, NHMD contains out-of-distribution examples. Hence, it is well suited for domain adaptation studies. The introduction of the NHMD dataset imposes better diversity coverage of vulnerable populations and generalisation for HMC tasks in a global public health surveillance setting. Additionally, we present a novel multi-task learning approach for HMC tasks by combining literal word meaning prediction as an auxiliary task. Experimental results demonstrate that the proposed approach outperforms state-of-the-art methods statistically significantly (p < 0.01, Wilcoxon test) in terms of F1 score over the state-of-the-art and shows that our new dataset poses a strong challenge to the existing HMC methods.

## Dataset
The splits for each category are shown beelow:

| Label  | Label Description | Train | Dev | Test
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0  | Health mention  | 923 | 112 | 113 |
| 1 | Other mention | 3,950 | 491 | 496 |
| 2  | Non-health mention  | 1,335 | 173 | 168 |
