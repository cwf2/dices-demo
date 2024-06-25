import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import os
import requests
import pickle

LOCAL_PATH = 'data'


#
# download data
#

def downloadData(local_path=LOCAL_PATH):
    '''Make sure data files are present locally'''
    local_path = 'data'
    remote_path = 'https://github.com/cwf2/dices-mta/raw/main/data/'
    data_files = ['merged.csv', 'mother-child.csv', 'mother_diction.csv']

    if not os.path.exists(local_path):
        os.mkdir(local_path)
    for filename in data_files:
        if not os.path.exists(os.path.join(local_path, filename)):
            print(f'downloading {filename}')
            res = requests.get(remote_path + filename, json={"download":""})
            if not res.ok:
                res.raise_for_status()
            with open(os.path.join(local_path, filename), 'wb') as f:
                f.write(res.content)
            
            
#
# process data
#

def processMothers(local_path=LOCAL_PATH):
    # tokens
    tokens_file = os.path.join(local_path, 'merged.csv')
    tokens = pd.read_csv(tokens_file, dtype=str)

    # mothers
    mothers_file = os.path.join(local_path, 'mother-child.csv')
    mothers = pd.read_csv(mothers_file, sep='\t')

    def motherValidation(rec):
        '''check whether any speaker-addressee combo is in the mother-child list'''
        valid_keys = list(mothers.spkr + ':' + mothers.addr)

        if rec['spkr'] is np.nan:
            return False
        if rec['addr'] is np.nan:
            return False
    
        for spkr in str(rec['spkr']).split(','):
            for addr in str(rec['addr']).split(','):
                key = f'{spkr}:{addr}'
                if key in valid_keys:
                    return True

        return False
    
    tokens['mother'] = tokens.apply(motherValidation, axis=1)
    return tokens


def addFeatures(tokens, local_path=LOCAL_PATH):
    # features
    class_file = os.path.join(local_path, 'mother_diction.csv')
    lemma_class = pd.read_csv(class_file)

    lem_dict = dict()
    for label in lemma_class.label.unique():
        if not pd.isna(label):
            lem_dict[label] = lemma_class.loc[lemma_class.label == label, 'lemma'].values
        
    for tag in ['family', 'pers_1s', 'pers_2s']:
        tokens[tag] = tokens['lemma_spacy'].isin(lem_dict[tag]) | tokens['lemma_cltk'].isin(lem_dict[tag])

    tokens['interrog'] = tokens['lemma_spacy'].isin(lem_dict['interrog'])
    tokens['pers'] = (tokens['pers_1s'] | tokens['pers_2s'])
    tokens['imper'] = (tokens['mood_cltk'] == 'imperative') | (tokens['mood_spacy'] == 'Imp')

    return tokens

#
# analysis
#

# log odds ranking

def rankFeatures(col, top=None):
    # freq in non-mother speeches
    freq_others = tokens.loc[~tokens['mother']].groupby(col).size().reset_index(name='count')
    freq_others['freq'] = freq_others['count'].div(freq_others['count'].sum())
    
    # freq in mother speeches
    freq_mother = tokens.loc[tokens['mother']].groupby(col).size().reset_index(name='count')
    freq_mother['freq'] = freq_mother['count'].div(freq_mother['count'].sum())

    # merge the two tables, so we have mother, non-mother freqs for each feature
    x = freq_others.merge(freq_mother, on=col, suffixes=('_others', '_mother'))

    # calculate log odds
    x['lod'] = np.log((x['freq_mother'] + 1) / (x['freq_others'] + 1))
    x = x.sort_values('lod', ascending=False)

    # optionally select just the top ranked results
    if top is not None:
        x = x[:top]

    # map the hand-picked feature classes onto the results where applicable
    x = x.merge(lemma_class.rename(columns={'lemma':col}), on=col, how='left') 
    
    return(x)
    
    
# rolling window samples

def rollingSamples(tokens):

    results = []
    for label, group in tokens.groupby('speech_id'):
        df = group.groupby('line_id', sort=False).agg(
            author = ('author', 'first'),
            work = ('work', 'first'),
            l_fi = ('l_fi', 'first'),
            l_la = ('l_la', 'first'),
            spkr = ('spkr', 'first'),
            addr = ('addr', 'first'),
            mother = ('mother', 'first'),
            tokens = ('token_spacy', 'count'),
            imper = ('imper', 'sum'),
            family = ('family', 'sum'),
            pers = ('pers', 'sum'),
            interrog = ('interrog', 'sum'),
        )
        results.append(
            pd.DataFrame(dict(
                speech_id = label,
                author = df['author'],
                work = df['work'],
                l_fi = df['l_fi'],
                l_la = df['l_la'],
                spkr = df['spkr'],
                addr = df['addr'],
                mother = df['mother'],
                lines = df['tokens'].rolling(window=5, min_periods=1, center=True).count(),
                tokens = df['tokens'].rolling(window=5, min_periods=1, center=True).sum(),
                family = df['family'].rolling(window=5, min_periods=1, center=True).sum(),
                imper = df['imper'].rolling(window=5, min_periods=1, center=True).sum(),
                pers = df['pers'].rolling(window=5, min_periods=1, center=True).sum(),
                interrog = df['interrog'].rolling(window=5, min_periods=1, center=True).sum(),
            ))
        )
    results = pd.concat(results)
    results['imper_norm'] = results['imper'].div(results['tokens'])
    results['family_norm'] = results['family'].div(results['tokens'])
    results['pers_norm'] = results['pers'].div(results['tokens'])
    results['interrog_norm'] = results['interrog'].div(results['tokens'])

    results['comp'] = results['imper'] + results['family'] + results['pers'] + results['interrog']
    results['comp_norm'] = results['comp'].div(results['tokens'])

    return results


# aggregate at speech level
def aggSamples(samples):
    x = samples.groupby('speech_id', sort=False).agg(
        author = ('author', 'first'),
        work = ('work', 'first'),
        l_fi = ('l_fi', 'first'),
        l_la = ('l_la', 'first'),
        spkr = ('spkr', 'first'),
        addr = ('addr', 'first'),
        mother = ('mother', 'first'),
        lines = ('tokens', 'count'),
        tokens = ('tokens', 'sum'),
        comp_avg = ('comp', 'mean'),
        comp_sum = ('comp', 'sum'),
        comp_max = ('comp', 'max'),
    )
    x['loc'] = x['l_fi'] + '-' + x['l_la']
    x['comp_norm'] = x['comp_sum'] / x['tokens']
    x = x.sort_values('comp_max', ascending=False)

    return x

#
# figures
#

def linePlot(results, speech_id):
    mask = results['speech_id']==speech_id
    ys = results.loc[mask, 'comp']

    _, l_fi = ys.index.values[0].rsplit(':', 1)
    _, l_la = ys.index.values[-1].rsplit(':', 1)
    author = results.loc[mask, 'author'].iloc[0]
    work = results.loc[mask, 'work'].iloc[0]
    title = f"{author} {work} {l_fi}-{l_la}"

    if '.' in l_fi:
        _, l_fi = l_fi.rsplit('.', 1)
    l_fi = int(l_fi)
    xs = np.arange(l_fi, l_fi+len(ys))

    avg = results['comp'].median()
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(xs, ys)
    ax.axhline(avg, color=sns.light_palette('#79C')[3], ls='--')
    ax.set_ylabel('rolling composite score')
    ax.set_xlabel('line')
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim((0,20))
    ax.set_title(title)
    plt.close(fig)
    return fig

def stackPlot(results, speech_id):
    mask = results['speech_id']==speech_id
    cols = ['family', 'imper', 'pers', 'interrog']
    df = results.loc[mask, cols]
    _, l_fi = df.index.values[0].rsplit(':', 1)
    _, l_la = df.index.values[-1].rsplit(':', 1)
    author = results.loc[mask, 'author'].iloc[0]
    work = results.loc[mask, 'work'].iloc[0]
    title = f"{author} {work} {l_fi}-{l_la}"

    if '.' in l_fi:
        _, l_fi = l_fi.rsplit('.', 1)
    l_fi = int(l_fi)
    xs = np.arange(l_fi, l_fi+len(df))
    df['line'] = xs

    avg = results['comp'].median()
    
    ax = df.plot.area(x='line', linewidth=0, figsize=(8,5))
    ax.axhline(avg, color='white', ls='--')
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim((0,20))
    ax.set_title(title)
    ax.set_ylabel('rolling score (5-line window)')
    ax.set_xlabel('line')
    fig = ax.figure
    plt.close(fig)
    return fig

def hl(col):
    return lambda string: f'<span style="font-weight:bold;color:{col}">{string}</span>'

def highlight(tokens, speech_id):
    mask = tokens['speech_id']==speech_id
    foo = pd.DataFrame(dict(
        line_id = tokens.loc[mask, 'line_id'],
        token = tokens.loc[mask, 'token_spacy'],
    ))
    foo.loc[tokens.loc[mask, 'family'], 'token'] = foo.loc[tokens.loc[mask, 'family'], 'token'].apply(hl('blue'))
    foo.loc[tokens.loc[mask, 'imper'], 'token'] = foo.loc[tokens.loc[mask, 'imper'], 'token'].apply(hl('orange'))
    foo.loc[tokens.loc[mask, 'pers'], 'token'] = foo.loc[tokens.loc[mask, 'pers'], 'token'].apply(hl('green'))
    foo.loc[tokens.loc[mask, 'interrog'], 'token'] = foo.loc[tokens.loc[mask, 'interrog'], 'token'].apply(hl('red'))
    
    html = '<table>' + '\n'.join(foo
        .groupby("line_id", sort=False)
        .agg(
            loc = ("line_id", lambda s: '<td>' + s.iloc[0].rsplit(':', 1)[1] + '</td>'),
            tokens = ("token", lambda s: '<td>' + ' '.join(s) + '<td>'),)
        .apply(lambda row: f'<tr>{row["loc"]}{row["tokens"]}</tr>', axis=1)
    ) + '</table>'
    
    return html

    
#
# User Experience
#
    
with st.status("Preparing data...", expanded=True) as status:
    st.write("Downloading remote data")
    downloadData()

    pickled_tokens = os.path.join(LOCAL_PATH, 'tokens.pickle')
    if os.path.exists(pickled_tokens):
        st.write("Loading cached token data")
        with open(pickled_tokens, 'rb') as f:
            tokens = pickle.load(f)
    else:
        st.write("Checking mother attribution")
        tokens = processMothers()
    
        st.write("Tagging hand-selected features")
        tokens = addFeatures(tokens)
        
        st.write("Caching")
        with open(pickled_tokens, 'wb') as f:
            pickle.dump(tokens, f)

    pickled_rolling = os.path.join(LOCAL_PATH, 'rolling.pickle')
    if os.path.exists(pickled_rolling):
        st.write("Loading cached rolling samples")
        with open(pickled_rolling, 'rb') as f:
            rolling = pickle.load(f)
    else:
        st.write("Calculating rolling samples")
        rolling = rollingSamples(tokens)
        
        st.write("Caching")
        with open(pickled_rolling, 'wb') as f:
            pickle.dump(rolling, f)
    
    pickled_aggregated = os.path.join(LOCAL_PATH, 'aggregated.pickle')
    if os.path.exists(pickled_aggregated):
        st.write("Loading cached aggregated samples")
        with open(pickled_aggregated, 'rb') as f:
            aggregated = pickle.load(f)
    else:
        st.write("Aggregating by speech")
        aggregated = aggSamples(rolling)
        
        st.write("Caching")
        with open(pickled_aggregated, 'wb') as f:
            pickle.dump(aggregated, f)

    status.update(expanded=False)

st.write(stackPlot(rolling, '3662'))