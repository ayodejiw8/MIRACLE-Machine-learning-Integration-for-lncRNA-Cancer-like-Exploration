import tarfile, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_lncrna_analysis(file_path):
    # 1. Extraction
    extract_dir = 'analysis_output'
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
        tsv_file = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) 
                    for f in filenames if f.endswith('.tsv')][0]

    # 2. Data Cleaning
    df = pd.read_csv(tsv_file, sep='\t', skiprows=1)
    df = df[~df['gene_id'].str.startswith('N_')].copy()

    # 3. Filtering & Top Genes
    lnc_df = df[df['gene_type'] == 'lncRNA'].copy()
    pc_df = df[df['gene_type'] == 'protein_coding'].copy()
    top_10 = lnc_df.sort_values(by='tpm_unstranded', ascending=False).head(10)

    # 4. Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tpm_unstranded', y='gene_name', data=top_10, palette='viridis')
    plt.title('Top 10 lncRNA Expression (TPM)')
    plt.savefig('top_lncrnas.png')

    plt.figure(figsize=(10, 6))
    comp_df = pd.concat([lnc_df, pc_df])
    comp_df['log_tpm'] = np.log10(comp_df['tpm_unstranded'] + 0.01)
    sns.boxplot(x='gene_type', y='log_tpm', data=comp_df)
    plt.title('Expression Distribution Comparison')
    plt.savefig('expression_dist.png')

    # 5. Export
    lnc_df.to_csv('lncrna_results.csv', index=False)
    print("Analysis complete. Visuals saved as PNGs and data saved to 'lncrna_results.csv'.")

# Execute
run_lncrna_analysis('gdc_download_20260304_200058.372139.tar.gz')
