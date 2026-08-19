# Patent Hallucination Evaluation

This repository contains the experimental code, question dataset, and summarized results used to evaluate **hallucination behavior in large language models (LLMs) on patent-based multiple-choice questions**.

The study compares **Llama** and **Qwen** across different patent-context conditions to examine how relevant, irrelevant, and misleading patent information affects model accuracy.

## Repository Structure

```text
.
├── data/
│   ├── Llama.pdf
│   └── Qwen.pdf
│
├── llama/
│   ├── Llama-3incorrect-1correct.py
│   ├── Llama-4-random-incorrect-patents.py
│   ├── Llama-Abstract-Claim.py
│   ├── Llama-DetailedDescription-3incorrect.py
│   └── Llama-NoContext.py
│
├── qwen/
│   ├── QWEN-3incorrect-1correct.py
│   ├── QWEN-4-random-incorrect-patents.py
│   ├── QWEN-Abstract-Claim.py
│   ├── QWEN-DetailedDescription-3incorrect.py
│   └── QWEN-NoContext.py
│
├── Question-Generation.py
└── questions_5000.csv
```

## Overview

The project evaluates Llama and Qwen on a set of **5,000 patent-based multiple-choice questions** under several controlled context conditions.

The goal is to determine how the information supplied to an LLM influences its ability to answer correctly and how misleading patent context can contribute to hallucination or incorrect reasoning.

## Experimental Conditions

Each model is evaluated under five primary conditions:

1. **No Context**
   The model answers the question without being provided patent information.

2. **Abstract + Claim**
   The model receives relevant patent abstract and claim information before answering.

3. **4 Patents — 1 Correct, 3 Incorrect**
   The model receives four patents, including the correct patent and three incorrect patents.

4. **Detailed Description — 1 Correct, 3 Incorrect**
   The model receives detailed patent-description information from four patents, including one correct and three incorrect patents.

5. **4 Incorrect Patents**
   The model receives four incorrect patents without the correct patent being included.

Experiments also compare model behavior **with and without an "IDK" option**, allowing the model to indicate that the provided information is insufficient rather than selecting one of the available answers.

## Question Generation

`Question-Generation.py` contains the pipeline used to generate the patent-based multiple-choice questions used throughout the experiments.

The generated evaluation dataset is stored in:

```text
questions_5000.csv
```

The same question set is used across the experimental conditions to allow direct comparison of model performance.

## Llama Experiments

The `llama/` directory contains the scripts used to run the experimental conditions on Llama.

| Script                                    | Condition                                                  |
| ----------------------------------------- | ---------------------------------------------------------- |
| `Llama-NoContext.py`                      | No patent context                                          |
| `Llama-Abstract-Claim.py`                 | Abstract and claim context                                 |
| `Llama-3incorrect-1correct.py`            | 1 correct + 3 incorrect patents                            |
| `Llama-DetailedDescription-3incorrect.py` | Detailed descriptions with 1 correct + 3 incorrect patents |
| `Llama-4-random-incorrect-patents.py`     | 4 incorrect patents                                        |

## Qwen Experiments

The `qwen/` directory contains the corresponding experimental scripts for Qwen.

| Script                                   | Condition                                                  |
| ---------------------------------------- | ---------------------------------------------------------- |
| `QWEN-NoContext.py`                      | No patent context                                          |
| `QWEN-Abstract-Claim.py`                 | Abstract and claim context                                 |
| `QWEN-3incorrect-1correct.py`            | 1 correct + 3 incorrect patents                            |
| `QWEN-DetailedDescription-3incorrect.py` | Detailed descriptions with 1 correct + 3 incorrect patents |
| `QWEN-4-random-incorrect-patents.py`     | 4 incorrect patents                                        |

Keeping the experimental structure consistent between the two models allows their performance and hallucination behavior to be compared directly.

## Results

The `data/` directory contains PDF summaries of the experimental results:

```text
data/
├── Llama.pdf
└── Qwen.pdf
```

Each PDF presents a chart summarizing model performance across the experimental conditions.

The charts compare:

* No Context
* Abstract + Claim
* 1 Correct + 3 Incorrect Patents
* Detailed Description with 1 Correct + 3 Incorrect Patents
* 4 Incorrect Patents
* Performance **with IDK**
* Performance **without IDK**

These summaries provide a direct comparison of how each context configuration affects model accuracy.

## Experimental Pipeline

```text
Patent Data
     ↓
Question Generation
     ↓
5,000 Multiple-Choice Questions
     ↓
Context Construction
     ↓
Llama / Qwen
     ↓
With IDK vs. Without IDK
     ↓
Model Responses
     ↓
Accuracy & Hallucination Analysis
```

## Research Objective

The primary objective of this project is to investigate how **context quality and relevance influence hallucination in LLMs answering patent-related questions**.

Specifically, the experiments examine whether models:

* Improve when relevant patent information is provided.
* Become less reliable when misleading or incorrect patents are introduced.
* Successfully identify the correct patent when it is mixed with irrelevant patents.
* Respond differently when given abstracts/claims versus detailed descriptions.
* Avoid incorrect answers when given the option to respond that they do not know.
* Exhibit different hallucination patterns across Llama and Qwen.

## Reproducibility

Individual experimental conditions can be run independently.

For example:

```bash
python llama/Llama-NoContext.py
```

or:

```bash
python qwen/QWEN-NoContext.py
```

Other experimental scripts can be executed in the same manner to reproduce the corresponding context conditions.

## Project Status

This repository contains the experimental pipeline and results used for an ongoing research study of hallucination and context sensitivity in large language models applied to patent-based question answering.
