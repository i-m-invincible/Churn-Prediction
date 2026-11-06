# Validation record

Validated on 6 November 2026 using Python 3.12 and the pinned runtime dependencies.

| Check | Observed result |
|---|---|
| Independent training execution | Completed all 15 CV fits, final fits, test evaluations and artifact/figure generation |
| Dataset | 7,043 unique customer IDs, 19 model inputs, 11 missing total-charge values |
| Split | 5,634 training / 1,409 test customers; disjoint IDs; membership recorded |
| Selection | XGBoost, by maximum mean training-only five-fold CV ROC-AUC |
| Selected model CV ROC-AUC | 0.8477 mean, 0.0106 standard deviation |
| Selected model test metrics | Precision 0.6689, recall 0.5294, F1 0.5910, ROC-AUC 0.8469 |
| Saved artifacts | Checksums and package versions verified; separate model/preprocessor agree with pipeline |
| Report consistency | Test metrics recalculated from predictions; served model probabilities match report; README table matches metrics.json |
| Fresh isolated environment | requirements-dev.txt and editable package installed; pip check found no broken requirements |
| Static checks | Ruff passes; staged diff has no whitespace errors |
| Automated tests | 39 passed in the fresh isolated environment |
| Streamlit interaction tests | Initial load, submission, missing charges, no phone/internet, EDA page and evaluation page passed |
| Runtime server | Streamlit started and /_stcore/health returned HTTP 200 |
| Browser screenshots | Not completed: Chromium launch was blocked by the execution environment's Unix-socket restrictions |

The CI workflow is included but has not run on GitHub before publication.
This is validation of the local implementation, not a claim of publication or a public deployment.
