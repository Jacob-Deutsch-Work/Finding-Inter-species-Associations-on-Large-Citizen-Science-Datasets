Paper explaining the statistics and more here: https://arxiv.org/pdf/2508.14259

# Finding-Inter-species-Associations-on-Large-Citizen-Science-Datasets
Usage. Set DB_FILE_NAME to the path of your input CSV. Additional user-configurable parameters are defined near the top of the script (approximately lines 1–30); adjust these as needed.

Input format. The program expects a CSV with the same schema as the provided example file. Please match the column names, order, and data types shown in that example.

Dependencies. The code relies on several external Python packages. If a required package is missing, the script will raise an import error indicating which dependency to install. Install the listed package (e.g., via pip install <package>) and re-run.

Computational notes. Consider testing first on a small subset of the data. However, quality of the results improve with larger datasets; results obtained from very small datasets may not be useful.

Contact. For questions, feel free to contact me at: jacobadeutschwork@gmail.com
