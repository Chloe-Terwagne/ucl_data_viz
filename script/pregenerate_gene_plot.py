import pandas as pd
import numpy as np
import datetime
from pycirclize import Circos
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import weighted
from skimage.color.rgb_colors import darkgray

# DISPLAY OPTIONS -----------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

# CONSTANTS ----------------------------------------------------------------------------------------------------------
DATE = str(datetime.date.today()).replace('-', '')
PLOT = False
SAVING_PLOT = True
SAVING_DF = False

PATH_VAR = '/outputs/step5_scoring_var/20250408_variant_df_mechanistic_clarity_annotated_predicted_rf_v6_precision_param_grid_gzoomed.csv'
PATH_GENE_PANEL = '/Users/terwagc/PycharmProjects/variant_prioritisation_cohorts/gene_selection_scripts/out_df/20240724_1468_genes_disease_selected.tsv'
PATH_HUMAN_CHR = "/Users/terwagc/PycharmProjects/variant_prioritisation_cohorts/gene_selection_scripts/input_df/hg38_chr.bed"
PATH_GENES = '/outputs/step2_master_df_unfiltered/20250324_master_gene_df.csv'

# FUNCTION ----------------------------------------------------------------------------------------------------------

def variant_to_plot_in_circos(df):
    print('\n--Variants with in repetitive region removed')
    df['rf_pred_proba_100'] = df['rf_pred_proba']*100
    df_filtered = df[df['is_repetitive'] == False]
    print('--Variants fo not pass_d4_qc removed')
    df_filtered_qced = df_filtered[df_filtered['pass_d4_qc'] == True]
    print("Number of unique genes after pass_d4_qc:", len(df_filtered_qced.gene_region_cleaned.unique()))
    print("--Variants not in GEL library removed")
    df_gel = df_filtered_qced[df_filtered_qced['variant_set']=='Stringent GEL']
    df_gel=df_gel.copy()
    df_gel['clvar_CLNSIG'] = df_gel['clvar_CLNSIG'].fillna('not annotated')
    df_gel['clvar_CLNSIG'] = df_gel['clvar_CLNSIG'].replace('not_provided', 'not annotated')
    df_gel = df_gel.sort_values(by=['chromosome', 'start_pos_hg38', 'variant_id', 'gene_region_cleaned'])

    print(df_gel.shape)
    print(df_gel.clvar_CLNSIG.value_counts())
    print("Number of unique genes:", len(df_gel.gene_region_cleaned.unique()))
    print("\n--Variants with rf_pred_proba <=0.9 removed")
    df_gel_90= df_gel[df_gel['rf_pred_proba']>0.9]
    print('df_gel_90.shape: ', df_gel_90.shape)
    print(df_gel_90.clvar_CLNSIG.value_counts())
    # print("Unique genes:", df_gel_90.gene_region_cleaned.unique())
    print("Number of unique genes:", len(df_gel_90.gene_region_cleaned.unique()))

    print("\n--Variants with rf_pred_proba > 0.9 removed and to be discovered")
    df_gel_90_discovery=df_gel_90[df_gel_90['clvar_CLNSIG'].isin([
                                          'Conflicting_classifications_of_pathogenicity',
                                          'not annotated',
                                          'Uncertain_significance'
                                      ])]

    print('df_gel_90_discovery', df_gel_90_discovery.shape[0])
    print(df_gel_90_discovery.clvar_CLNSIG.value_counts())
    return df_gel_90.gene_region_cleaned.unique(), df_gel, df_gel_90_discovery


df_genes = pd.read_csv(PATH_GENES, low_memory=False)
df_var = pd.read_csv(PATH_VAR, low_memory=False)

gene_list_gel90, df_gel, df_var_discovery = variant_to_plot_in_circos(df_var)
df_genes["chr"] = "chr"+df_genes["gtf_seqname"].astype(str)
gene_list_gel90= np.append(gene_list_gel90,'SON')
df_genes = df_genes[df_genes['GENE_SYMBOL'].isin(gene_list_gel90)]

print("df_genes---------")
print(df_genes.head())
print(df_genes.shape)

print("df_gel---------")
print(df_gel.head())
print(df_gel.shape)

# INNER TRACK RF Prediction Scatter
c_highlited='#FFD100'
c_scatter_track_ec, c_scatter_track_ew = 'white', 0.1

clnvar_color_dict = {
    'Not in ClinVar': '#7f8c8d',
    'Uncertain Significance': '#5dade2',
    'Conflicting Interpretation': '#f39c12',
    'Benign and Likely Benign': '#58d68d',
    'Likely Benign': '#82e0aa',
    'Likely Pathogenic': '#e74c3c',
    'Pathogenic and Likely Pathogenic': '#c0392b',
    'Pathogenic': '#ff4d4d'  # Only if you still have this key
}

clnvar_zorder_dict = {
    'Not in ClinVar': 0,
    'Uncertain Significance': 1,
    'Conflicting Interpretation': 2,
    'Benign and Likely Benign': 4,
    'Likely Benign': 3,
    'Likely Pathogenic': 6,
    'Pathogenic and Likely Pathogenic': 7,
    'Pathogenic': 8  # Only if present
}


import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns

# Optional: Use seaborn style for consistency
sns.set(style="white", rc={"axes.facecolor": "#0e0e1a"})
rename_CLNSIG = {
    'not annotated': 'Not in ClinVar',
    'Uncertain_significance': 'Uncertain Significance',
    'Conflicting_classifications_of_pathogenicity': 'Conflicting Interpretation',
    'Benign/Likely_benign': 'Benign and Likely Benign',
    'Likely_benign': 'Likely Benign',
    'Likely_pathogenic': 'Likely Pathogenic',
    'Pathogenic/Likely_pathogenic': 'Pathogenic and Likely Pathogenic',
}
df_gel['clvar_CLNSIG']=df_gel['clvar_CLNSIG'].map(rename_CLNSIG)
df_gel['clvar_CLNSIG']=df_gel['clvar_CLNSIG'].fillna('Not in ClinVar')
for gene in gene_list_gel90:
    gene_variants = df_gel[df_gel['gene_region_cleaned'] == gene].copy()
    gene_variants['x_val'] = range(len(gene_variants))

    print(f"\nGene {gene} ---------------------------")
    print(gene_variants.shape)
    print(gene_variants.clvar_CLNSIG.value_counts())
    print(gene_variants[['variant_id','clvar_CLNSIG','gene_region_cleaned','rf_pred' , 'rf_pred_proba' ,'rf_pred_proba_100' ,'x_val']].head(20))

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#0e0e1a')  # Full figure background

    ax = plt.gca()
    ax.set_facecolor('#0e0e1a')
    ax.tick_params(colors='white', which='both', width=0.6, length=4)  # thinner ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(
        axis='y',
        which='both',
        left=True,  #
        right=False,
        labelleft=True,
        color='white',  # tick color
        width=0.6,  # line width
        length=4  # length of the tick marks
    )
    # Minimalist spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)


    ax.set_title(f"{gene}", color='white', fontsize=16, weight='bold' ,pad=60)
    ax.set_xlabel("Genomic Position", color='white',labelpad=30)
    ax.set_ylabel("Loss-of-Function Prediction", color='white', labelpad=30)

    # Highlight high-prediction variants
    highlight_variants = gene_variants[gene_variants['rf_pred_proba'] > 0.9]
    y_vals_highlight = highlight_variants['rf_pred_proba'].to_numpy()
    x_vals_highlight = highlight_variants['x_val'].to_numpy()
    print('highlight_variants -----------------------------------------------------------------')
    print(highlight_variants[['variant_id','clvar_CLNSIG','gene_region_cleaned','rf_pred' , 'rf_pred_proba' ,'rf_pred_proba_100' ,'x_val']].head(20))
    plt.scatter(
        x_vals_highlight, y_vals_highlight,
        facecolors='none',  # hollow fill
        edgecolors=c_highlited,  # colored edge
        marker='o',
        s=78,
        lw=1,
        label='Novel Loss-of-Function (>0.9)',
        zorder=0
    )

    # Plot by clinical significance
    for clin_sig, z_order in clnvar_zorder_dict.items():
        gene_variants_clin = gene_variants[gene_variants['clvar_CLNSIG'] == clin_sig]
        y_vals = gene_variants_clin['rf_pred_proba'].to_numpy()
        x_vals = gene_variants_clin['x_val'].to_numpy()
        color = clnvar_color_dict[clin_sig]
        transparent_color = to_rgba(color, alpha=0.5)
        print('\nclin_sig',clin_sig)
        print('color',color)
        print('z_order',z_order)

        plt.scatter(
            x_vals, y_vals,
            c=[transparent_color],
            ec=color,
            marker='o',
            lw=0.7,
            s=25,
            zorder=z_order,
            vmin=-1, vmax=0,
            label=clin_sig  # fallback to original if not mapped

        )



    # Optional: Add legend
    legend = plt.legend(
        facecolor='#0e0e1a',
        edgecolor='whitesmoke',
        fontsize=10,
        title='Clinical Significance\n',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        labelspacing=0.75,       # space between labels
        borderaxespad=1.3       # space between legend and plot
    )

    legend.get_title().set_color('white')
    legend.get_title().set_fontsize(12)


    for text in legend.get_texts():
        text.set_color('white')
    plt.tight_layout()
    # Save as PNG (high quality)
    plt.savefig(f"gene_plots/{gene}_lof_plot.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

    # plt.show()

