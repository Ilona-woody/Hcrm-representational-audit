# Supplementary Materials: Representing Human Capabilities in Occupational Databases

**Paper:** Ilona Negro, "Representing Human Capabilities in Occupational Databases: An Embedding-Based Audit of O\*NET, ESCO and Vocational Standards"  
**Journal:** International Journal on Advances in Systems and Measurements, IARIA, vol. 19, no. 1&2, 2026  
**Author:** Ilona Negro, Doctoral Program of Applied Artificial Intelligence, Alma Mater Europaea, Maribor, Slovenia  
**Contact:** ilona.negro@almamater.si

---

## Overview

This repository contains the data, anchor sets, and analysis scripts used in the embedding-based representational audit reported in the paper. The study evaluates how three occupational representation systems — O\*NET Work Activities, ESCO skills, and national vocational training standards — encode four human-centred capability pillars derived from the Human-Centred Resilience Model (HCRM), across three occupations: electrician, dog groomer, and massage therapist.

---

## Repository Structure

```
/
├── README.md                                         This file
├── scripts/
│   ├── 4pillar_compareall.py                         Core embedding and similarity script
│   ├── compare_jobs.py                               Results aggregation script
│   ├── sensitivity_analysis.py                       Robustness and sensitivity analysis
│   └── descriptor_length.py                          Descriptor length statistics utility
├── data/
│   ├── Dog_Groomer_Representation_Comparison_final.xlsx
│   ├── Electrician_Representation_Comparison_final.xlsx
│   └── Massage_Therapist_Representation_Comparison_final.xlsx
└── esco/
    ├── domestic_electrician_ESCO_23_2_2026.pdf
    ├── animal_groomer_ESCO_23_2_2026.pdf
    └── massage_therapist_ESCO_23_2_2026.pdf
```

---

## Data Files

Each occupation has one Excel file containing four tabs:

### Tab: RPDC
National vocational training standards collected from six European countries: Austria, Belgium, France, Germany, Hungary, and the Netherlands. Each row represents one extracted competency statement or descriptor unit.

**Columns:**
- `Country` — country of origin
- `Occupation` — occupation label
- `Source title` — title of the source document
- `Issuing Authority` — issuing body or authority
- `Year` — year of document
- `Regulatory level` — national, regional, or not regulated
- `URL` — source URL
- `Original text` — original language text
- `EN translation` — English translation produced via DeepL

**Notes:**
- Blank rows serve as visual country separators and carry no data. They are filtered out automatically by the analysis script via a minimum two-word threshold applied to the EN translation column.
- Metadata fields (Source title, Issuing Authority, Regulatory level) may contain minor formatting artefacts from source document copying and are provided for provenance documentation purposes only. They do not affect the analysis.
- The Electrician corpus is substantially larger than the other two, reflecting the higher level of procedural specification in electrical installation training standards.

**Descriptor counts used in analysis:**

| Occupation | RPDC | O\*NET | ESCO |
|---|---|---|---|
| Dog Groomer | 161 | 41 | 46 |
| Electrician | 2337* | 41 | 41 |
| Massage Therapist | 180 | 41 | 53 |

*The paper reports 2341 for the Electrician RPDC corpus. The difference of 4 reflects minor variation in the two-word minimum filtering between the original analysis run and the current file version. This does not affect the reported similarity scores or the substantive findings.

### Tab: ONET_Activities
O\*NET Work Activities descriptors drawn from the O\*NET Database 30.2. The same 41 descriptors are used for all three occupations, reflecting the standardised nature of the O\*NET Work Activities taxonomy. Importance and level scores are included for reference but were not used in the similarity analysis.

### Tab: ESCO_Skills
ESCO skill and competence descriptors extracted from ESCO v1.2.1 (last update 10 December 2025, accessed 8 February 2026). Both essential and optional skills and competences are included. The Description column contains the full skill label used in the embedding analysis.

**ESCO Concept URIs:**
- Electrician (domestic electrician): http://data.europa.eu/esco/occupation/5dbb9cf0-b226-402c-a295-2f42ef05ff8b
- Dog Groomer (animal groomer): http://data.europa.eu/esco/occupation/5a940ae0-3fa4-40f9-8ca4-17bf792da243
- Massage Therapist: http://data.europa.eu/esco/occupation/9487f835-b5a9-4742-8ecf-35520911e28c

### Tab: Pillar_Anchors
The complete set of behavioural anchor statements used to operationalise the four HCRM capability pillars. The same 40 anchors (10 per pillar) are used across all three occupations and all representation systems.

**Pillars:**
- P1 — Cognitive–Technical Judgement
- P2 — Embodied–Sensorimotor Calibration
- P3 — Trust-Calibrated Interpersonal Engagement
- P4 — Situated Adaptive Decision-Making

Anchors were finalised prior to running the cross-occupation comparisons and were not tuned to individual corpora, ensuring that similarity scores reflect representational alignment rather than optimisation artefacts.

---

## ESCO Source Documents

The `/esco/` folder contains the individual ESCO occupation pages downloaded as PDFs (accessed 23 February 2026), showing the full skill and competence listings used for descriptor extraction. These are provided to document the exact source content used in the analysis.

---

## Scripts

### 4pillar_compareall.py
The core analysis script. For each occupation Excel file, it:
1. Loads descriptors from the RPDC, ONET_Activities, and ESCO_Skills tabs
2. Loads anchor statements from the Pillar_Anchors tab
3. Embeds all texts using the sentence-transformers model `all-MiniLM-L6-v2`
4. Computes cosine similarity between each descriptor and each anchor
5. Assigns descriptor-level similarity scores using maximum cosine similarity across sentence segments and anchor statements
6. Outputs summary statistics and top-15 descriptor matches per pillar

**Usage:**
```
py 4pillar_compareall.py OccupationName_Representation_Comparison_final.xlsx
```

**Output files:**
- `*_summary_allstats.xlsx` — mean, median, SD, min, max, p10, p90 per pillar–system combination
- `*_top15_matches.xlsx` — top 15 highest-similarity descriptors per pillar and system
- `*_all_descriptor_scores.xlsx` — full similarity scores for all descriptors

### compare_jobs.py
Aggregation script that combines summary files from all three occupations into combined output tables for cross-occupation comparison.

**Usage:**
```
py compare_jobs.py Dog_Groomer*_summary_allstats.xlsx Electrician*_summary_allstats.xlsx Massage_Therapist*_summary_allstats.xlsx
```

### sensitivity_analysis.py
Robustness check script implementing two sensitivity analyses reported in Section VII.F of the paper:
1. Mean vs maximum aggregation rule (using `all-MiniLM-L6-v2`)
2. Alternative embedding model (`all-mpnet-base-v2`) with maximum aggregation

**Usage:**
```
py sensitivity_analysis.py Dog_Groomer_Representation_Comparison_final.xlsx Electrician_Representation_Comparison_final.xlsx Massage_Therapist_Representation_Comparison_final.xlsx
```

**Output files:**
- `sensitivity_mean_vs_max.xlsx`
- `sensitivity_model_comparison.xlsx`
- `sensitivity_ranking_table.xlsx`
- `sensitivity_ranking_table.csv`

---

## Preprocessing Rules

The following preprocessing steps are applied by the analysis script:

1. **Whitespace normalisation** — non-breaking spaces replaced with standard spaces; multiple spaces collapsed to single space
2. **Minimum word threshold** — descriptors or segments with fewer than 2 words are excluded
3. **Sentence segmentation** — longer descriptors (primarily from RPDC) are split into sentence-level segments using punctuation and line-break boundaries. Similarity is computed at segment level; the maximum similarity across segments is retained as the descriptor-level score
4. **Translation** — non-English RPDC source documents were translated into English using DeepL prior to embedding. O\*NET and ESCO descriptors are natively in English
5. **ONET descriptor selection** — where both a full description and a short label are available, the full description is preferred

---

## Requirements

```
sentence-transformers
pandas
numpy
matplotlib
openpyxl
```

Install with:
```
pip install sentence-transformers pandas numpy matplotlib openpyxl
```

The first run will automatically download the embedding models from HuggingFace:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- `sentence-transformers/all-mpnet-base-v2` (~438MB, sensitivity analysis only)

---

## Reproducibility Note

The analysis is fully reproducible from the files in this repository. To reproduce the main results:

1. Clone this repository
2. Install requirements
3. Run `4pillar_compareall.py` for each occupation file
4. Run `compare_jobs.py` on the three summary output files
5. Run `sensitivity_analysis.py` for the robustness checks

Minor variations in similarity scores (typically in the fourth decimal place) may occur across different hardware or software versions due to floating-point precision differences in the embedding computation.

---

## Citation

If you use these materials, please cite:

I. Negro, "Representing Human Capabilities in Occupational Databases: An Embedding-Based Audit of O\*NET, ESCO and Vocational Standards," *International Journal on Advances in Systems and Measurements*, IARIA, vol. 19, no. 1&2, 2026.

---

## License

The data, anchor sets, and scripts in this repository are made available for research and reproducibility purposes. O\*NET data is a product of the U.S. Department of Labor and is in the public domain. ESCO data is copyright of the European Union and is made available under the EUPL licence. National vocational standard documents are the property of their respective issuing authorities and are reproduced here in extracted form for research purposes only.
