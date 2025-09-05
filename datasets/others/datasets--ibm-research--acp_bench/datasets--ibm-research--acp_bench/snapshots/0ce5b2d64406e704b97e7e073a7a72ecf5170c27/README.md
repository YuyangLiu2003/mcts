---
license: cdla-permissive-2.0
task_categories:
- question-answering
pretty_name: ACP Bench
tags:
- planning
- reasoning
dataset_info:
- config_name: acp_app_bool
  dataset_size: 223204
  download_size: 65362
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 182349
    num_examples: 130
  - name: validation
    num_bytes: 40855
    num_examples: 40
- config_name: acp_areach_bool
  dataset_size: 194445
  download_size: 48183
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 153012
    num_examples: 120
  - name: validation
    num_bytes: 41433
    num_examples: 40
- config_name: acp_just_bool
  dataset_size: 532461
  download_size: 112484
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 434458
    num_examples: 130
  - name: validation
    num_bytes: 98003
    num_examples: 40
- config_name: acp_land_bool
  dataset_size: 290859
  download_size: 75354
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 236580
    num_examples: 130
  - name: validation
    num_bytes: 54279
    num_examples: 40
- config_name: acp_prog_bool
  dataset_size: 234181
  download_size: 68746
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 192246
    num_examples: 130
  - name: validation
    num_bytes: 41935
    num_examples: 40
- config_name: acp_reach_bool
  dataset_size: 234074
  download_size: 65099
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 193493
    num_examples: 130
  - name: validation
    num_bytes: 40581
    num_examples: 40
- config_name: acp_val_bool
  dataset_size: 487661
  download_size: 108833
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 386961
    num_examples: 130
  - name: validation
    num_bytes: 100700
    num_examples: 40
- config_name: acp_app_mcq
  dataset_size: 326819
  download_size: 96360
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 262778
    num_examples: 130
  - name: validation
    num_bytes: 64041
    num_examples: 40
- config_name: acp_areach_mcq
  dataset_size: 284305
  download_size: 76059
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 219286
    num_examples: 120
  - name: validation
    num_bytes: 65019
    num_examples: 40
- config_name: acp_just_mcq
  dataset_size: 932999
  download_size: 197964
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 748097
    num_examples: 130
  - name: validation
    num_bytes: 184902
    num_examples: 40
- config_name: acp_land_mcq
  dataset_size: 341104
  download_size: 79190
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 272655
    num_examples: 130
  - name: validation
    num_bytes: 68449
    num_examples: 40
- config_name: acp_prog_mcq
  dataset_size: 331333
  download_size: 100358
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 265012
    num_examples: 130
  - name: validation
    num_bytes: 66321
    num_examples: 40
- config_name: acp_reach_mcq
  dataset_size: 327799
  download_size: 92620
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 264430
    num_examples: 130
  - name: validation
    num_bytes: 63369
    num_examples: 40
- config_name: acp_val_mcq
  dataset_size: 795585
  download_size: 160933
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - dtype: string
    name: query
  - dtype: string
    name: answer
  splits:
  - name: test
    num_bytes: 620792
    num_examples: 130
  - name: validation
    num_bytes: 174793
    num_examples: 40
- config_name: acp_app_gen
  dataset_size: 1066976
  download_size: 189054
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 887265
    num_examples: 130
  - name: validation
    num_bytes: 179711
    num_examples: 40
- config_name: acp_areach_gen
  dataset_size: 1044322
  download_size: 193224
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 870213
    num_examples: 130
  - name: validation
    num_bytes: 174109
    num_examples: 40
- config_name: acp_just_gen
  dataset_size: 1189965
  download_size: 192965
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    sequence:
      sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 964832
    num_examples: 130
  - name: validation
    num_bytes: 225133
    num_examples: 40
- config_name: acp_land_gen
  dataset_size: 1646526
  download_size: 251907
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    struct:
    - name: 'no'
      sequence: string
    - name: 'yes'
      sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 1443231
    num_examples: 130
  - name: validation
    num_bytes: 203295
    num_examples: 40
- config_name: acp_prog_gen
  dataset_size: 1017916
  download_size: 182814
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    struct:
    - name: neg
      sequence: string
    - name: pos
      sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 840693
    num_examples: 130
  - name: validation
    num_bytes: 177223
    num_examples: 40
- config_name: acp_nexta_gen
  dataset_size: 1250535
  download_size: 235381
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    struct:
    - name: maybe
      sequence: string
    - name: 'no'
      sequence: string
    - dtype: string
      name: opt
    - name: 'yes'
      sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 1062476
    num_examples: 130
  - name: validation
    num_bytes: 188059
    num_examples: 40
- config_name: acp_reach_gen
  dataset_size: 1013515
  download_size: 178491
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - name: answer
    sequence: string
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 842742
    num_examples: 130
  - name: validation
    num_bytes: 170773
    num_examples: 40
- config_name: acp_val_gen
  dataset_size: 1189899
  download_size: 194619
  features:
  - dtype: int64
    name: id
  - dtype: string
    name: group
  - dtype: string
    name: context
  - dtype: string
    name: question
  - dtype: int64
    name: answer
  - dtype: string
    name: PDDL_domain
  - dtype: string
    name: PDDL_problem
  splits:
  - name: test
    num_bytes: 990923
    num_examples: 130
  - name: validation
    num_bytes: 198976
    num_examples: 40

configs:
- config_name: acp_app_bool
  data_files:
  - split: test
    path: acp_app_bool/test.parquet
  - split: validation
    path: acp_app_bool/validation.parquet
- config_name: acp_app_mcq
  data_files:
  - split: test
    path: acp_app_mcq/test.parquet
  - split: validation
    path: acp_app_mcq/validation.parquet
- config_name: acp_areach_bool
  data_files:
  - split: test
    path: acp_areach_bool/test.parquet
  - split: validation
    path: acp_areach_bool/validation.parquet
- config_name: acp_areach_mcq
  data_files:
  - split: test
    path: acp_areach_mcq/test.parquet
  - split: validation
    path: acp_areach_mcq/validation.parquet
- config_name: acp_just_bool
  data_files:
  - split: test
    path: acp_just_bool/test.parquet
  - split: validation
    path: acp_just_bool/validation.parquet
- config_name: acp_just_mcq
  data_files:
  - split: test
    path: acp_just_mcq/test.parquet
  - split: validation
    path: acp_just_mcq/validation.parquet
- config_name: acp_land_bool
  data_files:
  - split: test
    path: acp_land_bool/test.parquet
  - split: validation
    path: acp_land_bool/validation.parquet
- config_name: acp_land_mcq
  data_files:
  - split: test
    path: acp_land_mcq/test.parquet
  - split: validation
    path: acp_land_mcq/validation.parquet
- config_name: acp_prog_bool
  data_files:
  - split: test
    path: acp_prog_bool/test.parquet
  - split: validation
    path: acp_prog_bool/validation.parquet
- config_name: acp_prog_mcq
  data_files:
  - split: test
    path: acp_prog_mcq/test.parquet
  - split: validation
    path: acp_prog_mcq/validation.parquet
- config_name: acp_reach_bool
  data_files:
  - split: test
    path: acp_reach_bool/test.parquet
  - split: validation
    path: acp_reach_bool/validation.parquet
- config_name: acp_reach_mcq
  data_files:
  - split: test
    path: acp_reach_mcq/test.parquet
  - split: validation
    path: acp_reach_mcq/validation.parquet
- config_name: acp_val_bool
  data_files:
  - split: test
    path: acp_val_bool/test.parquet
  - split: validation
    path: acp_val_bool/validation.parquet
- config_name: acp_val_mcq
  data_files:
  - split: test
    path: acp_val_mcq/test.parquet
  - split: validation
    path: acp_val_mcq/validation.parquet
- config_name: acp_app_gen
  data_files:
  - path: acp_app_gen/test.parquet
    split: test
  - path: acp_app_gen/validation.parquet
    split: validation
- config_name: acp_areach_gen
  data_files:
  - path: acp_areach_gen/test.parquet
    split: test
  - path: acp_areach_gen/validation.parquet
    split: validation
- config_name: acp_just_gen
  data_files:
  - path: acp_just_gen/test.parquet
    split: test
  - path: acp_just_gen/validation.parquet
    split: validation
- config_name: acp_land_gen
  data_files:
  - path: acp_land_gen/test.parquet
    split: test
  - path: acp_land_gen/validation.parquet
    split: validation
- config_name: acp_prog_gen
  data_files:
  - path: acp_prog_gen/test.parquet
    split: test
  - path: acp_prog_gen/validation.parquet
    split: validation
- config_name: acp_nexta_gen
  data_files:
  - path: acp_nexta_gen/test.parquet
    split: test
  - path: acp_nexta_gen/validation.parquet
    split: validation
- config_name: acp_reach_gen
  data_files:
  - path: acp_reach_gen/test.parquet
    split: test
  - path: acp_reach_gen/validation.parquet
    split: validation
- config_name: acp_val_gen
  data_files:
  - path: acp_val_gen/test.parquet
    split: test
  - path: acp_val_gen/validation.parquet
    split: validation

---

# ACP Bench 

<p align="center">
    <a href="https://ibm.github.io/ACPBench" target="_blank">🏠 Homepage</a>    •     
    <a href="https://doi.org/10.1609/aaai.v39i25.34857" target="_blank">📄 Paper</a> •     
    <a href="https://arxiv.org/abs/2503.24378" target="_blank">📄 Paper</a> 
</p>

ACPBench is a benchmark dataset designed to evaluate the reasoning capabilities of large language models (LLMs) in the context of Action, Change, and Planning. It spans 13 diverse domains:


* Blocksworld
* Logistics
* Grippers 
* Grid 
* Ferry
* FloorTile
* Rovers
* VisitAll
* Depot
* Goldminer
* Satellite
* Swap
* Alfworld

## Task Types in ACPBench

ACPBench includes the following 8 reasoning tasks:

1. Action Applicability (app)
2. Progression (prog)
3. Atom Reachability (reach)
4. Validation (val)
5. Action Reachability (areach)
6. Justification (just)
7. Landmarks (land)
8. Next Action (nexta)

## Task Formats

The first 7 tasks are available in:
* Boolean (yes/no) format
* Multiple-choice format
* Generative format

The Next Action task is provided only in generative format.

## Access

Development and test sets are available for download via:
* ACPBench GitHub Repository
* Hugging Face Dataset Hub


```
@inproceedings{KokelKSS25ACP
  author       = {Harsha Kokel and
                  Michael Katz and
                  Kavitha Srinivas and
                  Shirin Sohrabi},
  title        = {ACPBench: Reasoning about Action, Change, and Planning},
  booktitle    = {{AAAI}},
  publisher    = {{AAAI} Press},
  year         = {2025}
  url          = {https://doi.org/10.1609/aaai.v39i25.34857}
}
```

```
@misc{KokelKSS25ACPHard,
  title       = {ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning}, 
  author      = {Harsha Kokel and 
                 Michael Katz and 
                 Kavitha Srinivas and 
                 Shirin Sohrabi},
  year        = {2025},
  eprint      = {2503.24378},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url         = {https://arxiv.org/abs/2503.24378}, 
}
```