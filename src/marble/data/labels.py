import pandas as pd


def get_tcga_nsclc_slide_df(slide_df_path):
    """
    Load the TCGA NSCLC slide DataFrame from the given path.
    """
    df = pd.read_csv(slide_df_path)
    df = df[['slide_id', 'oncotree_code']]
    df.rename(columns={'oncotree_code': 'label'}, inplace=True)
    df['slide_id'] = df['slide_id'].str.removesuffix('.svs')

    return df


def get_panda_slide_df(slide_df_path, autogen_csv_path):
    """
    Load the PANDA slide DataFrame from the given path.
    """
    # Load the autogen CSV file
    processed_df = pd.read_csv(autogen_csv_path, low_memory=False)
    processed_df['slide_id'] = processed_df['slide_id'].str.removesuffix('.tiff')
    processed_df = processed_df[processed_df['status'] == 'already_exist']

    df = pd.read_csv(slide_df_path)
    df = df[df['image_id'].isin(processed_df['slide_id'])]
    df['slide_id'] = df['image_id'].copy()
    df['label'] = df['isup_grade'].copy()

    df = df[['slide_id', 'label']].reset_index(drop=True)

    return df


def get_tcga_survival_slide_df(
        csv_path,
        autogen_csv_path,
        label_col='survival_months',
        n_bins=4,
        eps=1e-6
    ):
    """
    Load the TCGA survival slide DataFrame from the given path.
    """
    # Load the autogen CSV file
    processed_df = pd.read_csv(autogen_csv_path, low_memory=False)
    processed_df['slide_id'] = processed_df['slide_id'].str.removesuffix('.svs')
    processed_df = processed_df[processed_df['status'] == 'already_exist']

    slide_data = pd.read_csv(csv_path, low_memory=False)
    slide_data = slide_data[slide_data['slide_id'].isin(processed_df['slide_id'])]
    if 'case_id' not in slide_data:
        slide_data.index = slide_data.index.str[:12]
        slide_data['case_id'] = slide_data.index
        slide_data = slide_data.reset_index(drop=True)

    if not label_col:
        label_col = 'survival_months'
    else:
        assert label_col in slide_data.columns

    patients_df = slide_data.drop_duplicates(['case_id']).copy()
    uncensored_df = patients_df[patients_df['censorship'] < 1]

    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = slide_data[label_col].max() + eps
    q_bins[0] = slide_data[label_col].min() - eps

    disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    patients_df.insert(2, 'label', disc_labels.values.astype(int))

    return patients_df
