import pandas as pd
import numpy as np
import datetime
from pycirclize import Circos
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from skimage.color.rgb_colors import darkgray
import matplotlib.gridspec as gridspec
import textwrap

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
    # print("Unique genes:", df_gel_90.gene_region_cleaned.unique())
    print("Number of unique genes:", len(df_gel_90.gene_region_cleaned.unique()))

    print("\n--Variants with rf_pred_proba > 0.9 removed and to be discovered")
    df_gel_90_discovery=df_gel_90[df_gel_90['clvar_CLNSIG'].isin([
                                          'Conflicting_classifications_of_pathogenicity',
                                          'not annotated',
                                          'Uncertain_significance'
                                      ])]

    print('df_gel_90_discovery', df_gel_90_discovery.shape[0])
    return df_gel_90.gene_region_cleaned.unique(), df_gel, df_gel_90_discovery


def print_gene_panel(df):
    print(df.head())
    print(df.shape[0])
    print(len(df['geneSymbol_panel_app_x'].unique()))
    # Using value_counts with NaNs included
    value_counts_with_nan = df['penetrance'].fillna('NaN').value_counts()
    value_counts_with_nan.index = value_counts_with_nan.index.where(value_counts_with_nan.index != 'NaN', None)
    # print(value_counts_with_nan)

    # Using value_counts with NaNs included
    value_counts_with_nan = df['diseaseGroup'].fillna('NaN').value_counts()
    value_counts_with_nan.index = value_counts_with_nan.index.where(value_counts_with_nan.index != 'NaN', None)
    # print(value_counts_with_nan)

    # Using value_counts with NaNs included
    value_counts_with_nan = df['diseaseSubgroup'].fillna('NaN').value_counts()
    value_counts_with_nan.index = value_counts_with_nan.index.where(value_counts_with_nan.index != 'NaN', None)
    # print(value_counts_with_nan)

    # Using value_counts with NaNs included
    value_counts_with_nan = df['panelID'].fillna('NaN').value_counts()
    value_counts_with_nan.index = value_counts_with_nan.index.where(value_counts_with_nan.index != 'NaN', None)
    # print(value_counts_with_nan)

    # Using value_counts with NaNs included
    value_counts_with_nan = df['panelName'].fillna('NaN').value_counts()
    value_counts_with_nan.index = value_counts_with_nan.index.where(value_counts_with_nan.index != 'NaN', None)
    # print(value_counts_with_nan)

    # DDG2P = ‘Severe undiagnosed neurodevelopmental disorder and/or congenital anomalies, abnormal growth parameters, dysmorphic features, and unusual behavioural phenotypes’.

    #
    # for panel_id in df["panelID"].unique():
    #     print("\n===== PANEL ID ", panel_id, '=========')
    #     subset_df= df[df["panelID"]==panel_id]
    #     print("Panel name: ",list(subset_df["panelName"].unique()))
    #     print("number unique disease group:\t", print(list(subset_df["diseaseGroup"].unique())))
    #     print("num unique disease subgroup:\t", print(list(subset_df["diseaseSubgroup"].unique())))


# LOAD DATA ----------------------------------------------------------------------------------------------------------

df_var = pd.read_csv(PATH_VAR, low_memory=False)
df_panel = pd.read_csv(PATH_GENE_PANEL, index_col=0, header=0, sep='\t')
df_genes = pd.read_csv(PATH_GENES)
df_human_chr = pd.read_csv(PATH_HUMAN_CHR, sep = '\t')
df_human_chr = df_human_chr.iloc[:24]

# PROCESSING ---------------------------------------------------------------------------------------------------------

gene_list_gel90, df_gel, df_var_discovery = variant_to_plot_in_circos(df_var)
df_genes["chr"] = "chr"+df_genes["gtf_seqname"].astype(str)
gene_list_gel90= np.append(gene_list_gel90,'SON')
df_genes = df_genes[df_genes['GENE_SYMBOL'].isin(gene_list_gel90)]

# PRINT HEADS FOR DEBUGGING ------------------------------------------------------------------------------------------

print("df_var_discovery---------")
print(df_var_discovery.head())
print(df_var_discovery.shape)
print("df_genes---------")
print(df_genes.head())
print(df_genes.shape)
print("df_panel---------")
print(df_panel.head())
print(df_panel.shape)

# VISUAL CONFIG ------------------------------------------------------------------------------------------------------
# plt.style.use('dark_background')

# Generate Circos Sectors from Chromosome Info
df_human_chr = df_human_chr[df_human_chr['#chrom']!='chrX']
df_human_chr.loc[df_human_chr['#chrom'] == 'chrY', 'chromEnd'] //= 4

buffer = 9500000
df_human_chr['chromEnd']+= buffer
transl = buffer/2
sectors = dict(zip(df_human_chr['#chrom'], df_human_chr['chromEnd']))
circos = Circos(sectors, space=1.4)

# Circos Plot Parameters ------------------------------------------------------------------------------------------------
x=1.8
gene_width_plot= 1.8
disease_track_heights=0.2
track_gene_on_chrom = (91,100)
track_gene_panel = (70.5,89)
track_rf_lim = (5,68)
t=track_gene_panel[1] # top starting

# colors ------------------------------------------------------------------------------------------------
# OUTER TRACK genes on chromosome
c_chr_background = (68/255, 70/255, 84/255, 0.55)
# c_gene = '#4cc9f0'
c_gene='#3ca6d8'
c_gene_ec, c_gene_ew = (68/255, 70/255, 84/255, 0.75), 1.4

# MIDDLE TRACK panel genes
c_gp1, c_gp2 = '#6f42c1', '#c5b358'
c_gp_track_ec, c_gp_track_ew = '#0e0e1a', 0.5

# INNER TRACK RF Prediction Scatter
c_highlited='#FFD100'
c_scatter_track_ec, c_scatter_track_ew = 'white', 0.1

clnvar_color_dict = {
    'not annotated': '#7f8c8d',  # Medium gray for visibility
    'Uncertain_significance': '#5dade2',  # Bright blue
    'Conflicting_classifications_of_pathogenicity': '#f39c12',  # Rich amber

    'Benign': '#2ecc71',  # Vivid green
    'Benign/Likely_benign': '#58d68d',  # Softer lime green
    'Likely_benign': '#82e0aa',  # Pastel green

    'Likely_pathogenic': '#e74c3c',  # Bright red
    'Pathogenic/Likely_pathogenic': '#c0392b',  # Deeper red
    'Pathogenic': '#ff4d4d'  # Neon-like red for emphasis
}


clnvar_zorder_dict = {
    'not annotated': 0,
    'Uncertain_significance': 1,
    'Conflicting_classifications_of_pathogenicity': 2,
    'Benign': 5,
    'Benign/Likely_benign': 4,
    'Likely_benign': 3,
    'Likely_pathogenic': 6,
    'Pathogenic/Likely_pathogenic': 7,
    'Pathogenic': 8
}
disease_group=[("Neurology and neurodevelopmental disorders", t), ("Dysmorphic and congenital abnormality syndromes",t-x), ("Tumour syndromes",t-2*x),
               ("Metabolic disorders",t-3*x), ("Cancer Programme",t-4*x), ("Skeletal disorders",t-5*x),
               ("Viral research",t-6*x), ("Endocrine disorders",t-7*x), ("Haematological disorders",t-8*x),
               ("Hearing and ear disorders",t-9*x), ("Renal and urinary tract disorders",t-10*x), ("Cardiovascular disorders",t-11*x),
               ("Gastroenterological disorders",t-12*x),("Rheumatological disorders",t-13*x),("Respiratory disorders",t-14*x),("Ciliopathies",t-15*x),("Haematological and immunological disorders",t-16*x),("Growth disorders",t-17*x)]
disease_group_abbr = {
    "Neurology and neurodevelopmental disorders": "NND",
    "Dysmorphic and congenital abnormality syndromes": "DCAS",
    "Tumour syndromes": "TS",
    "Metabolic disorders": "MS",
    "Cancer Programme": "CP",
    "Skeletal disorders": "SD",
    "Viral research": "VR",
    "Endocrine disorders": "ED",
    "Haematological disorders": "HD",
    "Hearing and ear disorders": "HED",
    "Renal and urinary tract disorders": "RUTD",
    "Cardiovascular disorders": "CVD",
    "Gastroenterological disorders": "GastroD",
    "Rheumatological disorders": "RD",
    "Respiratory disorders": "RespD",
    "Ciliopathies": "C",
    "Haematological and immunological disorders": "HID",
    "Growth disorders": "GD"
}

# Main Circos Loop ------------------------------------------------------------------------------------------------------
circos.line(r=track_gene_panel[0]-1, color="darkgray", lw=1)
j = 0
print("HERE -----------------------------------")
for sector in circos.sectors:
    if(sector.name != "chrX") and  sector.name != "chrY":
        chr_nb = sector.name.replace("chr", "")
        sector.text(chr_nb, size=10, r=104, color=c_gene, weight='bold')
        sector.line(r=track_gene_on_chrom[0], color=c_gene, lw=1, zorder=4)
        # OUTER TRACK genes on chromosome ------------------------------------
        track_chrom = sector.add_track((track_gene_on_chrom[0], track_gene_on_chrom[1]))
        track_chrom.axis(fc=c_chr_background, lw=0)
        print("\n CHROMOS:", chr_nb,'*/')
        # Filter rows for the current sector (chromosome)
        sector_genes = df_genes[df_genes["chr"] == sector.name].sort_values("gtf_start")
        # for _, row in sector_genes.iterrows():
            # print("    <div class=\"hover-zone-"+row['GENE_SYMBOL']+"\"></div>")

        for _, row in sector_genes.iterrows():
            # print("."+row['GENE_SYMBOL']+"_lof_plot,", end=' ')
            print(row['GENE_SYMBOL'], '\t', row['gtf_start'])

            # print(".hover-zone-"+row['GENE_SYMBOL']+" {position: absolute; cursor: pointer; width: 0.3%; height: 3%; background-color: green; z-index: 2;\ntop: 17%;\nleft: 29.7%;\ntransform: rotate(2deg);\n}")
            midpoint = (row['gtf_end'] + row['gtf_start']) / 2
            start = midpoint - buffer/2
            end = midpoint + buffer/2
            track_chrom.rect(transl+start, transl+end, color=c_gene, ec=c_gene_ec, lw=c_gene_ew)

        # MIDDLE TRACK panel genes------------------------------------
        for disease in disease_group[0:10]:
            disease_df = df_panel[df_panel['diseaseGroup']==disease[0]]
            matched_genes = sector_genes[sector_genes["GENE_SYMBOL"].isin(disease_df["GENE_SYMBOL"])]

            track = sector.add_track((disease[1]-x+0.5, disease[1]))
            color = c_gp1 if j % 2 == 0 else c_gp2
            # track.axis(fc="white", lw=0)
            j += 1

            # add y label on the gene panel tracks
            if sector.name == "chr1":
                track.yticks([15], [disease_group_abbr[disease[0]]], vmin=10, vmax=20, side="left", label_size=3.5,
                              line_kws=dict(color="red", lw=0),
                              label_margin=-1,
                              text_kws=dict(color=color, weight='bold'))

            for _, row in matched_genes.iterrows():
                midpoint = (row['gtf_end'] + row['gtf_start']) / 2
                start = midpoint - buffer / 2
                end = midpoint + buffer / 2
                track.rect(transl + start, transl + end, color=color, ec=c_gp_track_ec, lw=c_gp_track_ew)

        # INNER TRACK RF Prediction Scatter------------------------------------
        track_rf_proba = sector.add_track(track_rf_lim)
        chr_variants = df_gel[df_gel['chromosome'] == int(chr_nb)].copy()
        chr_variants['x_val'] = np.linspace(sector.start, sector.end, len(chr_variants))

        for clin_sig, z_order in clnvar_zorder_dict.items():
            clin_variants = chr_variants[chr_variants['clvar_CLNSIG'] == clin_sig]
            y_vals = -clin_variants['rf_pred_proba'].to_numpy()
            x_vals = clin_variants['x_val'].to_numpy()
            color = clnvar_color_dict[clin_sig]

            # track_rf_proba.scatter(
            #     x_vals, y_vals,
            #     c=color, ec=color,
            #     marker='o', lw=c_scatter_track_ew, s=6.5,
            #     alpha=0.45, zorder=z_order,
            #     vmin=-1, vmax=0
            # )
            transparent_color = to_rgba(color, alpha=0.45)  # semi-transparent face
            track_rf_proba.scatter(
                x_vals, y_vals,
                c=transparent_color,  # transparent fill
                ec=color,  # solid edge
                marker='o',
                lw=0.4,
                s=7.5,
                zorder=z_order,
                vmin=-1, vmax=0
            )

        highlight_variants = chr_variants[chr_variants['rf_pred_proba'] > 0.9]
        y_vals_highlight = -highlight_variants['rf_pred_proba'].to_numpy()
        x_vals_highlight = highlight_variants['x_val'].to_numpy()
        track_rf_proba.scatter(
            x_vals_highlight, y_vals_highlight,
            c=c_highlited, marker='o', s=7.5, ec=darkgray,
            lw=0, vmin=-1, vmax=0,
            label='High RF Prediction (>0.9)', zorder=5
        )
    # dark navy


# Clean version of explanatory text with manual wrapping
wrapped_text = (
    "Rare diseases affect over 350 million people worldwide, yet 70% of patients\nremain undiagnosed—even after whole-genome sequencing (WGS)\n"
    "Why? While WGS identifies many genetic variants, we often lack the tools to\ndetermine which of these changes are truly responsible for disease. Without\n"
    "a clear understanding of the functional effects of these variants,it's difficult\nto link them to clinical symptoms—leaving many patients undiagnosed.\n\n"
    "Our approach combines pooled prime editing (a state-of-the-art genome-editing\ntechnique) with machine learning to prioritize, functionally test and\ninterpret thousands of variants directly in human cell lines.\n\n"
    "    • We prioritized over 21,000 variants, including 8,000 known control variants,\n    from WGS data of rare disease participants.\n"
    "    • Each variant was tested for its impact on cell viability over time.\n"
    "    • We build a supervised model trained on control variants to predict\n    Loss-of-Function variants based on experimental results.\n\n"
    "From 11,440 variants in the Genomics England cohort, we identified \n492 high-confidence loss-of-function variants—offering new avenues\nfor rare disease diagnosis and personalized medicine.\n\n\n\n"

)

legend_title = "Circos plot of predicted impact of rare disease-associated variants tested by pooled prime editing.\n\n"
legend_text = ("The outermost track displays human chromosomes and highlights genomic regions where variants were tested. Genes targeted for variant testing are represented by blue bars, positioned according to their chromosomal location. Each gene included in the analysis\n"
                "is associated with at least one clinical phenotype and is mapped to a specific disease group. Disease group membership is visualized as concentric colored rings, where each ring corresponds to a clinical panel. From the outer to the inner ring, the number \n"
                "of genes in each disease group is as follows: 123 in Neurology and Neurodevelopmental Disorders (NND), 60 in Dysmorphic and Congenital Abnormality Syndromes (DCAS), 26 in Tumour Syndromes (TS), 35 in Metabolic Disorders (MS), 26 in the Cancer Programme (CP),\n"
                "29 in Skeletal Disorders (SD), 23 in Viral Research (VR), 21 in Endocrine Disorders (ED), 17 in Haematological Disorders (HD), and 14 in Hearing and Ear Disorders (HED). For example, the outermost purple ring highlights genes associated with NND. The innermost\n"
                "track is a scatterplot where each dot represents a tested genetic variant, positioned according to its genomic coordinate indexes. Variant impact was quantified using a supervised classifier trained on control variants. Variants are plotted radially: the more\n"
                "inward the dot, the higher its predicted loss-of-function (LoF) score. A total of 492 high-confidence LoF variants were identified and are highlighted in bright yellow, indicating strong predicted functional effects with potential relevance to rare disease phenotypes.")

# Let pycirclize handle its own figure and axis
fig = circos.plotfig()
fig.patch.set_facecolor('#0e0e1a')

# Now manually add a new axes to the right for the text
text_ax = fig.add_axes([1.05, 0.1, 0.3, 0.8])  # [left, bottom, width, height]
text_ax.axis('off')
up=-0.05
# Title line – BIGGER and BOLD
text_ax.text(0.05, 1.03+up, "Zeroing in on disease-causing DNA variants\n\n",
             fontsize=17,
             fontweight='bold',
             color='white',
             ha='left',
             va='top',
             fontname='Helvetica')

# Add body text below header
text_ax.text(0.05, 0.94+up, wrapped_text,
             fontsize=14,
             color='#7f8c8d',
             ha='left',
             va='top',
             fontname='Helvetica',
             linespacing=1.5)

# text_ax.text(0.05, 0.1+up, legend_title,
#              fontsize=11.5,
#              fontweight='bold',
#              color='white',
#              ha='left',
#              va='top',
#              fontname='Helvetica')
#
# # Add body text below header
# text_ax.text(0.05, 0.06+up, legend_text,
#              fontsize=9.5,
#              color='#7f8c8d',
#              ha='left',
#              va='top',
#              fontname='Helvetica',
#              linespacing=1.5)



# Title
# fig.suptitle("Zeroing in on Disease-Causing Human DNA Variants",
#              fontsize=25,
#              fontweight='bold',
#              color='#7f8c8d',
#              ha='center',
#              x=0.5,   # Center horizontally across the figure
#              y=1.02,  # Adjust vertically if it overlaps
#              fontname='Helvetica')


fig.text(0.98, 1.1, "Understanding DNA changes at scale to improve rare disease diagnosis",
         fontsize=32,
         fontweight='heavy',
         color='white',
         ha='center',
         va='top',
         fontname='Helvetica')

text_ax.text(0.05, 0.00, "Hover over the genes in chromosomes 1 to 11 to explore their variant scores.",
             fontsize=14,
             fontweight='bold',
             color=c_gene,
             ha='left',
             va='top',
             fontname='Helvetica')

fig.text(0, -0.01, legend_title,
             fontsize=11.5,
             fontweight='bold',
             color='white',
             ha='left',
             va='top',
             fontname='Helvetica')

# Add body text below header
fig.text(0, -0.04, legend_text,
             fontsize=9.5,
             color='#7f8c8d',
             ha='left',
             va='top',
             fontname='Helvetica',
             linespacing=1.5)


fig.tight_layout()
fig.savefig("main-image.png",
            dpi=1000,
            bbox_inches='tight',
            pad_inches=0.8,
            facecolor='#0e0e1a')

fig.show()

