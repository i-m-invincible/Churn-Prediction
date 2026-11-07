# Streamlit Community Cloud deployment

## Deploy

1. Sign in at https://share.streamlit.io/ and connect the GitHub account that owns the repository.
2. Choose **Create app → Deploy a public app from GitHub**.
3. Set repository `i-m-invincible/Churn-Prediction`, branch `main`, and main file `app.py`.
4. Open **Advanced settings** and select **Python 3.12**. No secrets are needed.
5. Deploy. The repository is public; confirm Sharing is set to **This app is public and searchable**.
6. Open the resulting URL in a signed-out browser and test all three workspaces.

Sign-in can require accepting Streamlit's terms and authorizing its GitHub integration.
Review the permissions in the provider's consent screen. No passwords, tokens or
credentials belong in this repository or in chat.

Python is selected in Community Cloud's deployment dialog; `.python-version` is
a local development hint, not a replacement for that setting. If an existing
deployment uses a different Python version, follow Streamlit's documented
redeployment process rather than changing the model's required package versions.

## Verify the public deployment

- The initial example shows an actual model probability and a rendered factor chart.
- Submit a different profile. Test missing total charges and no phone/internet services.
- Load both example profiles and confirm that the estimates change.
- Open Explore Data, change the grouping, and inspect the original EDA figures.
- Open Model Performance; check the comparison, ROC/calibration plots and confusion matrix.
- Download the example CSV, upload it, and download predictions for all three rows.
- Test an empty or malformed CSV and confirm a readable error without an internal traceback.
- Confirm a visitor can use the application without signing in.
- After adding the tested URL and actual dashboard screenshot to README, test again
  following the final commit and automatic Cloud refresh.

Do not advertise a guessed subdomain or treat a sign-in screen, successful build,
or healthy HTTP endpoint alone as proof that the interactive public app works.

## Artifact integrity and Windows line endings

The original `reference_customer.json` is 495 bytes with LF line endings. A
Windows checkout with `core.autocrlf=true` converted it to 516 bytes with 21 CRLF
endings, which changed its raw SHA-256 even though its values were unchanged.

The fix has three parts:

- `.gitattributes` keeps text checkouts at LF and preserves the dataset/model bytes.
- Training writes reference JSON as UTF-8 with explicit LF endings on every OS.
- Reference verification normalizes only CRLF to LF before hashing. Values,
  punctuation, spaces and every other byte remain significant. A real change is
  rejected before any model file is deserialized. Binary artifacts are always
  checked byte for byte.

`metadata.json` records the rule and the same expected artifact digests. The
trained model, preprocessing, reference values and evaluation results were not
changed or retrained for this deployment work.

If a checksum fails for an actual modified/corrupt artifact, restore the trusted
artifact set together or run the documented offline training pipeline. Do not
delete expected hashes or set them to unverified values to silence the error.

## Runtime and resource choices

- Runtime: Python 3.12, with pinned packages in `requirements.txt`.
- Linux x86_64 uses `xgboost-cpu==3.1.2`; other platforms use the full matching wheel.
  Both expose the same Python API. The saved model is already CPU-trained.
- The served model is cached with at most one resource entry. Reports and
  training-only exploration data use a bounded data cache.
- Uploads: 5 MB / 10,000 rows. User uploads are not cached or written to the repository.
- No training, external data downloads, paid services, custom JavaScript or
  external frontend frameworks run at startup.
- Visitor errors are generic, with detailed diagnostics reserved for maintainer logs.
  XSRF/CORS protections stay enabled. Downloaded customer IDs are escaped against
  spreadsheet formula interpretation.

Community Cloud can hibernate inactive applications and its resource limits can
change. A cold start or wake-up is normal; it does not require retraining.

Official references:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python
- https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
