# Third-party provenance and notices

## Dataset

`data/telco.csv` is an unmodified copy of IBM's Telco Customer Churn CSV:

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Source revision: `d5371f5d83a446ad5673cbcca3b814b926491f8a`
- Source path: `data/Telco-Customer-Churn.csv`
- The source repository carries Apache License 2.0; its full text is retained in
  [third_party/IBM_APACHE_2.0.txt](third_party/IBM_APACHE_2.0.txt).
- No dataset-specific license is asserted beyond the notices provided by that source.
  The project MIT license applies to the independently implemented application code,
  not to a relicensing of IBM's dataset or third-party packages.

## Conceptual inspiration

Ibrahim Khan's MIT-licensed [Customer Churn Prediction project](https://github.com/ibrahimkhan-data/Customer-Churn-Prediction)
was reviewed for conceptual inspiration. Its application, notebook, training code,
UI styling and trained artifacts were not copied into this implementation.
Its original MIT notice is retained in [third_party/IBRAHIM_MIT.txt](third_party/IBRAHIM_MIT.txt)
as a provenance record. The previous implementation and its original notices are
also preserved in the separate full-history backup.

The new source code was developed for Bishal Dutta with AI assistance. Git commit
identity records the repository maintainer; it does not imply that no tools were used.
Python dependencies retain their own licenses.
