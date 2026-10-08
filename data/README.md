# Local benchmark data

Source: https://www.unitime.org/uct_datasets.php

Download the Fall Mathematics & Statistics instance in the **Purdue C8-2007 Data Sets** section:
https://www.unitime.org/data/josh/pu-fal07-ms.zip

Extract `pu-fal07-ms.xml` into this directory. The matching source has creation timestamp `Thu Feb 14 22:17:36 EST 2008`. Do not substitute the older identically named download in the page's separate Fall 2007 section.

Run from the project root:

```powershell
python data_extraction_purdue.py --xml data/pu-fal07-ms.xml --output-dir data/processed
```

This source reproduces all 13 supplied tables in content. The package includes local XML and processed CSVs, but Git excludes them. Use Git to publish, rather than uploading the entire package through GitHub's web uploader. Third-party data are not covered by a project code license.
