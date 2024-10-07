#This Dashboard uses multiple tools for comparing and analyzing any number of deep learning model embeddings of the same molecules and their molecular properties
#The analysis shows the global and local organization of the model
#Run the prepare_data file to generate the csv files necessary for running the dashboard
#The dashboard will not work without those csv files

#import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, ctx, no_update

import base64
import textwrap
from io import BytesIO

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import statistics
from random import sample 
from operator import itemgetter
from collections import Counter

import glob
import os

#Download and generate all datasets needed to run the Dash app
#Define global variables

dir_name = os.path.dirname(os.path.realpath(__file__))

#Model relationships dataframe
model_relationships_file = glob.glob(dir_name+r'/data/relationships/*.csv')
general_model_relationships = [pd.read_csv(model_relationships_file[x]) for x in range(len(model_relationships_file))]

#EMBEDDINGS
embedding_files = sorted(glob.glob(dir_name+r'/data/embeddings/*.csv'))
general_model_embeddings = [pd.read_csv(embedding_files[x]) for x in range(len(embedding_files))]

#NAMING AND INDEXING LISTS AND DICTIONARIES
model_names = [embedding_files[x][embedding_files[x].rindex('/')+1:embedding_files[x].rindex('.')] for x in range(len(embedding_files))]
model_name_index = dict(zip(model_names,list(range(len(model_names)))))
model_index_name = dict(zip(list(range(len(model_names))),model_names))

#global variables for chosen dataset part 1
model_relationships = general_model_relationships[0].copy() if len(general_model_relationships) > 0 else None
model_embeddings = general_model_embeddings.copy()
all_molecules = model_embeddings[0]['SMILES'].tolist()
molecule_name_index = dict(zip(all_molecules,list(range(len(all_molecules)))))
molecule_index_name = dict(zip(list(range(len(all_molecules))),all_molecules))
molecules = all_molecules.copy()

#MOLECULAR PROPERTIES
feature_files = glob.glob(dir_name+r'/data/features/*.csv')
general_mol_features = pd.read_csv(feature_files[0],low_memory=False)

#SIMILARITIES
general_model_similarities = dict(zip(model_names,[pd.DataFrame(cosine_similarity(model_embeddings[x].set_index('SMILES'),(model_embeddings[x].set_index('SMILES')))).rename(columns=dict(zip(range(3282),all_molecules)),index=dict(zip(range(3282),all_molecules))) for x in range(len(model_embeddings))]))

#UMAPS
umap_files = [glob.glob(dir_name+r'/data/prepared_data/UMAPs/*umap_' + model_names[x] + r'.csv')[0] for x in range(len(model_names))]
general_model_umaps = dict(zip(model_names,[pd.read_csv(umap_files[x]).iloc[:,1:] for x in range(len(model_names))]))

#SPEARMAN
spearman_files = glob.glob(dir_name+r'/data/prepared_data/pairwise_spearman*.csv')
general_pairwise_spearmanr = pd.read_csv(spearman_files[0]).iloc[:,1:]

#NEAREST NEIGHBORS
overlap_files = glob.glob(dir_name+r'/data/prepared_data/overlap_func*.csv')
general_model_overlaps = pd.read_csv(overlap_files[0]).set_index('num_nbrs')

overlap_melted_files = glob.glob(dir_name+r'/data/prepared_data/overlap_func_melted*.csv')
general_model_overlaps_melted = pd.read_csv(overlap_melted_files[0]).iloc[:,1:]

nearest_neighbors_files = glob.glob(dir_name+r'/data/prepared_data/nearest_neighbors*.csv')
general_nearest_nbrs = pd.read_csv(nearest_neighbors_files[0])

#PROPERTY CORRELATIONS AND PREDICTIONS
property_correlations_files = glob.glob(dir_name+r'/data/prepared_data/property_correlations*.csv')
general_property_correlations = pd.read_csv(property_correlations_files[0]).set_index('Unnamed: 0')
property_prediction_files = glob.glob(dir_name+r'/data/prepared_data/property_predictions_r2_scores*.csv')
general_property_predictions = pd.read_csv(property_prediction_files[0]).set_index('Unnamed: 0').transpose()

#global variables for chosen dataset part 2
mol_features = general_mol_features.copy()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_mol_features = mol_features.select_dtypes(include=numerics)
numeric_mol_features = numeric_mol_features.loc[:, (numeric_mol_features != 0).any(axis=0)]
features = list(numeric_mol_features.columns)
features_index = dict(zip(features,list(range(len(features)))))

model_similarities = general_model_similarities.copy()

model_umaps = general_model_umaps.copy()

pairwise_spearmanr = general_pairwise_spearmanr.copy()

model_overlaps = general_model_overlaps.copy()
model_overlaps_melted = general_model_overlaps_melted.copy()
nearest_nbrs = general_nearest_nbrs.copy()

property_correlations = general_property_correlations.copy()
property_difference_index = dict(zip(property_correlations.columns,list(range(len(property_correlations.columns)))))
property_predictions = general_property_predictions.copy()
property_index = dict(zip(property_predictions.columns,list(range(len(property_predictions.columns)))))
corrs = property_correlations.copy()
scores = property_predictions.copy()
corrs.columns = scores.columns
c = corrs.melt().rename(columns={"Unnamed: 0":"property","value":"corrs"})
c['model'] = [model_index_name[c.index[x] % len(model_names)] for x in range(len(c.index))]
s = scores.melt().rename(columns={"Unnamed: 0":"property","value":"scores"})
s['model'] = [model_index_name[s.index[x] % len(model_names)] for x in range(len(s.index))]
corr_pred = c.merge(s,on=["property","model"])

good_corr = -0.3
bad_corr = -0.3
good_pred = 0.7
bad_pred = 0.7

corr_pred['bucket_model'] = ''
corr_pred['bucket'] = ''

for row in corr_pred.index:
    if(corr_pred.loc[row,'corrs']<good_corr)&(corr_pred.loc[row,'scores']>good_pred):
        corr_pred.loc[row,'bucket_model'] = 'good corr, good pred: ' + corr_pred.loc[row,'model']
        corr_pred.loc[row,'bucket'] = 'good corr, good pred'
    elif(corr_pred.loc[row,'corrs']<good_corr)&(corr_pred.loc[row,'scores']<bad_pred):
        corr_pred.loc[row,'bucket_model'] = 'good corr, bad pred: ' + corr_pred.loc[row,'model']
        corr_pred.loc[row,'bucket'] = 'good corr, bad pred'
    elif(corr_pred.loc[row,'corrs']>bad_corr)&(corr_pred.loc[row,'scores']>good_pred):
        corr_pred.loc[row,'bucket_model'] = 'bad corr, good pred: ' + corr_pred.loc[row,'model']
        corr_pred.loc[row,'bucket'] = 'bad corr, good pred'
    elif(corr_pred.loc[row,'corrs']>bad_corr)&(corr_pred.loc[row,'scores']<bad_pred):
        corr_pred.loc[row,'bucket_model'] = 'bad corr, bad pred: ' + corr_pred.loc[row,'model']
        corr_pred.loc[row,'bucket'] = 'bad corr, bad pred'

#PROPERTY TRUTH VALUES
property_truth_files = glob.glob(dir_name+r'/data/truth/*.csv')
property_truths_names = [property_truth_files[x][property_truth_files[x].rindex('/')+1:property_truth_files[x].rindex('.')] for x in range(len(property_truth_files))]
property_truths_name_file = dict(zip(property_truths_names,[pd.read_csv(property_truth_files[x]).set_index('SMILES').iloc[:,[-1]] for x in range(len(property_truths_names))]))


#GLOBAL DATAFRAMES
model1, model2, model1_similarity, model2_similarity = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
embedding_smiles1, embedding_smiles2 = pd.DataFrame(), pd.DataFrame()

selected_data, clicked_data = pd.DataFrame(), pd.DataFrame()

pairwise_data1 = pd.DataFrame()
pairwise_data2 = pd.DataFrame()

fig = px.scatter()
smiles_col = "SMILES"
show_img = True
svg_size = 200
alpha = 1
mol_alpha = 1
title_col = "SMILES"
show_coords = True
wrap = True
wraplen = 20
width = 150
fontfamily = "Arial"
fontsize = 12
reaction = False

df_data = embedding_smiles1.copy().reset_index()

colors = {0: "black"}

svg_height = svg_size
svg_width = svg_size


if isinstance(smiles_col, str):
    smiles_col = [smiles_col]

app = Dash()

layout = [
    html.H1(children='Dashboard for Analyzing and Comparing Multiple Deep Learning Model Embeddings', style={'textAlign':'center'}),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Compare: Stats and UMAPs', children=[
            dcc.Dropdown(model_names, model_names[0], id='dropdown-model1',style={'display':'inline-block','width':'49%'}),
            dcc.Dropdown(model_names, model_names[1], id='dropdown-model2',style={'display':'inline-block','width':'49%'}),
            dcc.Graph(id='model-similarity',style={'display':'inline-block','width':'50%'}),
            dcc.Graph(id='model-overlap',style={'display':'inline-block','width':'50%'}),
            html.Br(),
            html.Label('Color UMAPs by:'),
            html.Br(),
            dcc.Dropdown(['None','similarity'] + property_truths_names + features, 'None', id='dropdown-color'),
            html.Br(),
            html.Label('Click on a molecule to show its nearest neighbors or select a region of molecules to highlight the same molecules on the other UMAP. Coloring by similarity refers to cosine similarity distance for each molecule with respect to a clicked molecule',style={'display':'inline-block','vertical-align':'middle','text-align':'middle'}),
            html.Label('Toggle outliers:',style={'display':'inline-block','vertical-align':'middle','text-align':'middle','marginLeft':'50px'}),
            dcc.RadioItems(['On','Off'],'Off',id='toggle-outliers',style={'display':'inline-block','vertical-align':'middle'}),
            html.Br(),
            dcc.Graph(id='model1-umap',style={'display':'inline-block','width':'50%','height':'600px'}, clear_on_unhover=True),
            dcc.Graph(id='model2-umap',style={'display':'inline-block','width':'50%','height':'600px'}, clear_on_unhover=True),
        ]),
        dcc.Tab(label='Summary: Nearest neighbor heatmaps', children=[
            html.Label('number of neighbors'),
            dcc.Slider(id='num-nbrs', min=min(model_overlaps.index), max=max(model_overlaps.index), step=None,
                        marks={x:str(x) for x in model_overlaps.index},
                        value=model_overlaps.index[int(statistics.median(range(len(model_overlaps.index))))]),
            dcc.Graph(id='kneighbor-map',style={'height':'750px'}),
        ]),
        dcc.Tab(label='Compare: ∆Property vs. ∆Similarity', children=[
            dcc.Dropdown(model_names, model_names[0], id='dropdown-scatter-model1',style={'display':'inline-block','width':'49%'}),
            dcc.Dropdown(model_names, model_names[1], id='dropdown-scatter-model2',style={'display':'inline-block','width':'49%'}),
            dcc.Dropdown(property_truths_names + features, features[0], id='dropdown-feature'),
            html.Br(),
            html.Label('Click on a molecule pair to highlight other pairs including one of the two molecules on both graphs'),
            html.Br(),
            dcc.Graph(id='property-similarity-scatter1',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
            dcc.Graph(id='property-similarity-scatter2',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
        ]),
        dcc.Tab(label='Compare: Property descriptor correlations vs. predictions', children=[
            dcc.Dropdown(model_names, model_names[0], id='dropdown-corr-pred-scatter-model1',style={'display':'inline-block','width':'49%'}),
            dcc.Dropdown(model_names, model_names[1], id='dropdown-corr-pred-scatter-model2',style={'display':'inline-block','width':'49%'}),
            dcc.Graph(id='property-correlation-scatter',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
            dcc.Graph(id='property-prediction-scatter',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
        ]),
        dcc.Tab(label='Summary: Property descriptor correlations vs. predictions', children=[
            dcc.Graph(id='pairwise-property-correlation-heatmap',style={'display':'inline-block','width':'49%','height':'700px'}),
            dcc.Graph(id='pairwise-property-prediction-heatmap',style={'display':'inline-block','width':'49%','height':'700px'}),
            dcc.Graph(id='prediction-vs-correlation-scatter-properties',style={'display':'inline-block','height':'500px','width':'49%'}),
            dcc.Graph(id='prediction-vs-correlation-scatter-models',style={'display':'inline-block','height':'500px','width':'49%'}),
            html.Label('good corr threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='-0.3',id='input-good-corr',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('bad corr threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='-0.3',id='input-bad-corr',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('good pred threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='0.7',id='input-good-pred',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('bad pred threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='0.7',id='input-bad-pred',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            dcc.Graph(id='prediction-vs-correlation-buckets',style={'height':'400px'}),
            dcc.Graph(id='prediction-vs-correlation-scatter-bar',style={'height':'400px'}),
        ]),
        dcc.Tab(label='Property descriptor flow process', children=[
            dcc.Dropdown(['None'] + model_names, model_names[0], id='dropdown-flow-model1',style={'display':'inline-block','width':'49%'}),
            dcc.Dropdown(['None'] + model_names, model_names[1], id='dropdown-flow-model2',style={'display':'inline-block','width':'49%'}),
            html.Br(),
            html.Label('Select \'None\' for one model to view the flow of properties through model relationships starting from the other model',style={'display':'inline-block','vertical-align':'middle','text-align':'middle'}),
            html.Br(),
            html.Br(),
            html.Label('good corr threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='-0.3',id='flow-input-good-corr',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('bad corr threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='-0.3',id='flow-input-bad-corr',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('good pred threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='0.7',id='flow-input-good-pred',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('bad pred threshold',style={'display':'inline-block','width':'8%','vertical-align':'middle','text-align':'middle','marginLeft':'5%'}),
            dcc.Input(value='0.7',id='flow-input-bad-pred',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Br(),
            html.Br(),
            html.Label("start bucket",style={'display':'inline-block','width':'100px','vertical-align':'middle','text-align':'middle','marginLeft':'50px'}),
            dcc.Dropdown(["Any","good corr, good pred","bad corr, good pred","good corr, bad pred","bad corr, bad pred"], "good corr, good pred", id='dropdown-start',style={'display':'inline-block','vertical-align':'middle','width':'200px'}),
            html.Label("end bucket",style={'display':'inline-block','width':'100px','vertical-align':'middle','text-align':'middle','marginLeft':'50px'}),
            dcc.Dropdown(["Any","good corr, good pred","bad corr, good pred","good corr, bad pred","bad corr, bad pred"], "good corr, good pred", id='dropdown-end',style={'display':'inline-block','vertical-align':'middle','width':'200px'}),
            html.Br(),
            dcc.Graph(id='property-corr-pred-scatter',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
            dcc.Graph(id='property-corr-pred-sankey',style={'display':'inline-block','height':'600px','width':'49%'}, clear_on_unhover=True),
        ]),
        dcc.Tab(label='Model connections network maps', children=[
            html.Label('Spearman correlation threshold to form edge:',style={'display':'inline-block','width':'15%','vertical-align':'middle','text-align':'middle','marginLeft':'8%'}),
            dcc.Dropdown(['<','>'],'>',id='dropdown-operator-spearman',placeholder='operator',style={'display':'inline-block','vertical-align':'middle'}),
            dcc.Input(value='0',placeholder='spearman',id='input-spearman',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('Nearest ' + str(model_overlaps.index[int(statistics.median(range(len(model_overlaps.index))))]) + ' neighbors overlap threshold to form edge:',style={'display':'inline-block','width':'15%','vertical-align':'middle','text-align':'middle','marginLeft':'8%'}),
            dcc.Dropdown(['<','>'],'>',id='dropdown-operator-kneighbor',placeholder='operator',style={'display':'inline-block','vertical-align':'middle'}),
            dcc.Input(value='0',placeholder='percent overlap',id='input-kneighbor',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Label('Correlation of correlations threshold to form edge:',style={'display':'inline-block','width':'15%','vertical-align':'middle','text-align':'middle','marginLeft':'8%'}),
            dcc.Dropdown(['<','>'],'>',id='dropdown-operator-correlation',placeholder='operator',style={'display':'inline-block','vertical-align':'middle'}),
            dcc.Input(value='0',placeholder='correlation',id='input-correlation',style={'display':'inline-block','width':'5%','vertical-align':'middle'}),
            html.Br(),
            dcc.Graph(id='network-graph-spearman',style={'display':'inline-block','width':'33%','height':'600px'}),
            dcc.Graph(id='network-graph-kneighbor',style={'display':'inline-block','width':'33%','height':'600px'}),
            dcc.Graph(id='network-graph-correlation',style={'display':'inline-block','width':'33%','height':'600px'}),
        ]),
        dcc.Tab(label='Tree of model relationships', children=[
            dcc.Graph(id='model-relationships-treemap',style={'display':'inline-block'}),
        ]),
    ]),
    dcc.Tooltip(
        id="graph-tooltip", background_color=f"rgba(255,255,255,{alpha})"
    ),
    dcc.Store(id='data',data=pd.DataFrame().to_json(orient="split")),
    dcc.Store(id='threshold-data',data=pd.DataFrame().to_json(orient="split")),
]

app.layout = html.Div(children=layout)

@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("model1-umap", "hoverData"),
    Input("model2-umap", "hoverData"),
    Input("property-similarity-scatter1", "hoverData"),
    Input("property-similarity-scatter2", "hoverData"),
)
def display_hover(hoverData_umap1, hoverData_umap2, hoverData_scatter1, hoverData_scatter2):
    global embedding_smiles1, pairwise_data1
    
    if hoverData_umap1 is None and hoverData_umap2 is None and hoverData_scatter1 is None and hoverData_scatter2 is None :
        return False, no_update, no_update
    
    if hoverData_umap1 is not None or hoverData_umap2 is not None: 
        df_data = embedding_smiles1.copy().reset_index()
    elif hoverData_scatter1 is not None or hoverData_scatter2 is not None:
        df_data = pairwise_data1.copy().reset_index()


    if hoverData_umap1 is not None:
        pt = hoverData_umap1["points"][0]
    elif hoverData_umap2 is not None:
        pt = hoverData_umap2["points"][0]
    elif hoverData_scatter1 is not None:
        pt = hoverData_scatter1["points"][0]
    elif hoverData_scatter2 is not None:
        pt = hoverData_scatter2["points"][0]
    
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    curve_num = pt["curveNumber"]

    df_row = df_data.iloc[num]

    if hoverData_scatter1 is not None or hoverData_scatter2 is not None: 
        chosen_smiles = [molecule_index_name[df_row['variable']],molecule_index_name[df_row['variable2']]]
    else:
        value = smiles_col
        chosen_smiles = value
        

    hoverbox_elements = []

#modify show_img and title_col to be able to show two images/titles for two molecules
    if show_img:
        for col in chosen_smiles:
            if hoverData_scatter1 is not None or hoverData_scatter2 is not None: 
                smiles = col
            else: 
                smiles = df_row[col]

            buffered = BytesIO()
            if isinstance(smiles, str):
                # Generate 2D SVG if smiles column is a string

                d2d = rdMolDraw2D.MolDraw2DSVG(svg_width, svg_height)
                opts = d2d.drawOptions()
                opts.clearBackground = False
                if reaction:
                    try:
                        d2d.DrawReaction(ReactionFromSmarts(smiles, useSmiles=True))
                    except:
                        d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
                else:
                    d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
                d2d.FinishDrawing()
                img_str = d2d.GetDrawingText()
                buffered.write(str.encode(img_str))
                img_str = base64.b64encode(buffered.getvalue())
                img_str = f"data:image/svg+xml;base64,{repr(img_str)[2:-1]}"

            elif isinstance(smiles, Mol):
                # if smiles column is a Mol object, use the 3D coordinates of the mol object
                img = Chem.Draw.MolToImage(smiles)
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue())
                img_str = "data:image/png;base64,{}".format(repr(img_str)[2:-1])

            else:
                raise TypeError(
                    "smiles_col or mol_col not specified with the correct type."
                )
            if len(smiles_col) > 1:
                hoverbox_elements.append(
                    html.H2(
                        f"{col}",
                        style={
                            "color": colors[curve_num],
                            "font-family": fontfamily,
                            "fontSize": fontsize + 2,
                        },
                    )
                )
            hoverbox_elements.append(
                html.Img(
                    src=img_str,
                    style={
                        "width": "100%",
                        "background-color": f"rgba(255,255,255,{mol_alpha})",
                    },
                )
            )

    if title_col is not None:
        if hoverData_scatter1 is not None or hoverData_scatter2 is not None: 
            title = chosen_smiles[0] + " and " + chosen_smiles[1]        
        else:
            title = df_row[col]

        if len(title) > wraplen:
            if wrap:
                title = textwrap.fill(title, width=wraplen)
            else:
                title = title[:wraplen] + "..."

        # TODO colorbar color titles
        hoverbox_elements.append(
            html.H4(
                f"{title}",
                style={
                    "color": colors[curve_num],
                    "font-family": fontfamily,
                    "fontSize": fontsize,
                },
            )
        )
    if show_coords:
        x_label = 'x'
        y_label = 'y'
        hoverbox_elements.append(
            html.P(
                f"{x_label}: {pt['x']}",
                style={
                    "color": "black",
                    "font-family": fontfamily,
                    "fontSize": fontsize,
                },
            )
        )
        hoverbox_elements.append(
            html.P(
                f"{y_label} : {pt['y']}",
                style={
                    "color": "black",
                    "font-family": fontfamily,
                    "fontSize": fontsize,
                },
            )
        )
    children = [
        html.Div(
            hoverbox_elements,
            style={
                "width": f"{width}px",
                "white-space": "normal",
            },
        )
    ]

    return True, bbox, children

@callback(
        Output('data','data'),
        Input('dropdown-model1', 'value'),
        Input('dropdown-model2', 'value'),
)
def update_data(value1, value2):
    global molecules, model1_similarity, model2_similarity, model1, model2, model_similarities, molecules

    model1 = model_embeddings[model_name_index[value1]].set_index('SMILES').loc[molecules,:]
    model2 = model_embeddings[model_name_index[value2]].set_index('SMILES').loc[molecules,:]
    
    model1_similarity = model_similarities[value1].loc[molecules,molecules]
    model2_similarity = model_similarities[value2].loc[molecules,molecules]
    model1_similarity.columns = model1.index
    model2_similarity.columns = model2.index

    return pd.DataFrame().to_json(orient="split")

@callback(
    Output('model-similarity', 'figure'),
    Output('model-overlap', 'figure'),
    Input('data','data'),
    State('dropdown-model1', 'value'),
    State('dropdown-model2', 'value'),
)
def update_graph(data,value1,value2):
    global pairwise_spearmanr,model_name_index,model_overlaps

    row_index = np.intersect1d(np.where(pairwise_spearmanr['model1']==model_name_index[value1]),np.where(pairwise_spearmanr['model2']==model_name_index[value2]))

    if str(value1 + ', ' + value2) in model_overlaps.columns:
        fig_overlap = px.line(model_overlaps.drop(str(value1 + ', ' + value2),axis=1),title='% Overlap in Nearest Neighbors vs Number of Nearest Neighbors Between Two Models').update_layout(yaxis_title='percent overlap')
        fig_overlap.update_traces(line_color="gray",showlegend = False)
        fig_overlap.add_trace(go.Scattergl(y=list(model_overlaps[str(value1 + ', ' + value2)]),x=list(model_overlaps.index),name=str(value1 + ', <br>' + value2),mode='lines',line=dict(color="#ff0000")))
    else:
        fig_overlap = px.line(model_overlaps.drop(str(value2 + ', ' + value1),axis=1),title='% Overlap in Nearest Neighbors vs Number of Nearest Neighbors Between Two Models').update_layout(yaxis_title='percent overlap')
        fig_overlap.update_traces(line_color="gray",showlegend = False)
        fig_overlap.add_trace(go.Scattergl(y=list(model_overlaps[str(value2 + ', ' + value1)]),x=list(model_overlaps.index),name=str(value2 + ', <br>' + value1),mode='lines',line=dict(color="#ff0000")))
    

    fig_overlap.update_layout(legend_traceorder="reversed")

    return px.histogram(pairwise_spearmanr.iloc[row_index,2:].iloc[0,:],nbins=200,range_x=[-1, 1],title="Spearman Correlation Comparing Cosine Similarities of Molecules Between Two Models").update_layout(showlegend=False,xaxis_title='Spearman\'s rank correlation coefficient',yaxis_title='Number of molecules'),fig_overlap

@callback(
    Output('model1-umap','figure'),
    Output('model2-umap','figure'),
    Input('data','data'),
    State('dropdown-model1','value'),
    State('dropdown-model2','value'),
    State('dropdown-color','value')
)
def obtain_umaps(data,value1,value2,color_property):
    global model_umaps,embedding_smiles1,embedding_smiles2,model1_similarity,model2_similarity,molecules
    embedding_smiles1 = model_umaps[value1].set_index('SMILES').loc[molecules,:]
    embedding_smiles1.insert(2,'similarity',pd.DataFrame(model1_similarity.iloc[:,0]))
    embedding_smiles2 = model_umaps[value2].set_index('SMILES').loc[molecules,:]
    embedding_smiles2.insert(2,'similarity',pd.DataFrame(model2_similarity.iloc[:,0]))

    fig1 = px.scatter(embedding_smiles1,x='x',y='y',color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value1 + " Embeddings")
    fig2 = px.scatter(embedding_smiles2,x='x',y='y',color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value2 + " Embeddings")

    if color_property == 'None':
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")

    return fig1, fig2

@callback(
    Output('model1-umap', 'figure',allow_duplicate=True),
    Output('model2-umap', 'figure',allow_duplicate=True),
    Output('model1-umap','clickData'),
    Output('model2-umap','clickData'),
    State('dropdown-model1', 'value'),
    State('dropdown-model2', 'value'),
    Input('dropdown-color','value'),
    Input('toggle-outliers','value'),
    prevent_initial_call=True,
)
def update_umap_color(value1, value2, color_property,outliers):
    input_id = ctx.triggered_id

    global embedding_smiles1, embedding_smiles2, molecule_name_index, clicked_data, nearest_nbrs, property_truths_names, property_truths_name_file

    if color_property == 'similarity':
        embedding_smiles1 = model_umaps[value1].set_index('SMILES').loc[molecules,:]
        embedding_smiles1.insert(2,'similarity',pd.DataFrame(model1_similarity.iloc[:,0]))
        embedding_smiles2 = model_umaps[value2].set_index('SMILES').loc[molecules,:]
        embedding_smiles2.insert(2,'similarity',pd.DataFrame(model2_similarity.iloc[:,0]))
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',color=embedding_smiles1.iloc[:,-1].name,title="UMAP of " + value1 + " Embeddings",color_continuous_scale='haline')
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',color=embedding_smiles2.iloc[:,-1].name,title="UMAP of " + value2 + " Embeddings",color_continuous_scale='haline')
        return fig1, fig2, None, None
    elif (color_property in property_truths_names) and (len(property_truths_name_file[color_property].index.intersection(model_similarities[value1].index))>0):
        print(property_truths_name_file[color_property].index.intersection(model_umaps[model_names[0]].set_index('SMILES').index))
        property_column = property_truths_name_file[color_property]
        embedding_smiles1 = embedding_smiles1.drop(columns=embedding_smiles1.columns[2])
        embedding_smiles2 = embedding_smiles2.drop(columns=embedding_smiles2.columns[2])
        for smiles in property_column.index:
            if smiles in embedding_smiles1.index:
                embedding_smiles1.loc[smiles,color_property] = property_column.loc[smiles,property_column.columns[0]]
        for smiles in property_column.index:
            if smiles in embedding_smiles2.index:
                embedding_smiles2.loc[smiles,color_property] = property_column.loc[smiles,property_column.columns[0]]
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value1 + " Embeddings")
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value2 + " Embeddings")
        return fig1, fig2, None, None
    elif color_property in features:
        property_column = mol_features.set_index('SMILES').loc[molecules,:].loc[:,[color_property]]
        if outliers == 'Off':
            property_column[(np.abs(stats.zscore(property_column)) > 3).all(axis=1)] = np.nan
        embedding_smiles1 = embedding_smiles1.drop(columns=embedding_smiles1.columns[2])
        embedding_smiles1.insert(2,color_property,pd.DataFrame(property_column))
        embedding_smiles2 = embedding_smiles2.drop(columns=embedding_smiles2.columns[2])
        embedding_smiles2.insert(2,color_property,pd.DataFrame(property_column))
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value1 + " Embeddings")
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline',title="UMAP of " + value2 + " Embeddings")
        return fig1, fig2, None, None
    else:
        embedding_smiles1 = embedding_smiles1.drop(columns=embedding_smiles1.columns[2])
        embedding_smiles2 = embedding_smiles2.drop(columns=embedding_smiles2.columns[2])
        embedding_smiles1.loc[:,color_property] = np.nan
        embedding_smiles2.loc[:,color_property] = np.nan
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")
        return fig1, fig2, None, None

@callback(
    Output('model1-umap', 'figure',allow_duplicate=True),
    Output('model2-umap', 'figure',allow_duplicate=True),
    Output('model1-umap','selectedData'),
    Output('model2-umap','selectedData'),
    Input('model1-umap','clickData'),
    Input('model2-umap','clickData'),
    State('dropdown-model1', 'value'),
    State('dropdown-model2', 'value'),
    State('dropdown-color','value'),
    prevent_initial_call=True,
)
def update_clicked_data(clickData1, clickData2, value1, value2, color_property):
    input_id = ctx.triggered_id

    global embedding_smiles1, embedding_smiles2, molecule_name_index, clicked_data, nearest_nbrs

    fig1 = px.scatter(embedding_smiles1,x='x',y='y',color=embedding_smiles1.iloc[:,-1].name,title="UMAP of " + value1 + " Embeddings",color_continuous_scale='haline')
    fig2 = px.scatter(embedding_smiles2,x='x',y='y',color=embedding_smiles2.iloc[:,-1].name,title="UMAP of " + value2 + " Embeddings",color_continuous_scale='haline')

    if (color_property == 'None') or ((color_property in property_truths_names) and (len(property_truths_name_file[color_property].index.intersection(model_similarities[value1].index))==0)):
            fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
            fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")
    
    if clickData1 is not None or clickData2 is not None:
        if input_id == 'model1-umap':
            pt = clickData1["points"][0]['pointIndex']
            
            clicked_data = embedding_smiles1.reset_index().loc[pt]
            data_row = clicked_data.squeeze()
        elif input_id == 'model2-umap':
            pt = clickData2["points"][0]['pointIndex']
            
            clicked_data = embedding_smiles2.reset_index().loc[pt]
            data_row = clicked_data.squeeze()

        molecule = str(data_row['SMILES'])

        if color_property == 'similarity':
            embedding_smiles1.iloc[:,-1] = model1_similarity.loc[:,molecule]
            embedding_smiles2.iloc[:,-1] = model2_similarity.loc[:,molecule]
            
        fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings",color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline')
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings",color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline')

        if (color_property == 'None') or ((color_property in property_truths_names) and (len(property_truths_name_file[color_property].index.intersection(model_similarities[value1].index))==0)):
            fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
            fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")

        molecules1 = nearest_nbrs.iloc[molecule_name_index[molecule],[x + 10*model_name_index[value1] for x in list(range(10))]]
        molecules2 = nearest_nbrs.iloc[molecule_name_index[molecule],[x + 10*model_name_index[value2] for x in list(range(10))]]
        mol_smiles1 = [molecule_index_name[molecules1[x]] for x in range(len(molecules1))]
        mol_smiles2 = [molecule_index_name[molecules2[x]] for x in range(len(molecules2))]
        fig1.add_trace(go.Scattergl(x=list(embedding_smiles1.loc[mol_smiles1,'x'].astype(float)),y=list(embedding_smiles1.loc[mol_smiles1,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="Red"),
                    name="nearest neighbors"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=list(embedding_smiles2.loc[mol_smiles2,'x'].astype(float)),y=list(embedding_smiles2.loc[mol_smiles2,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="Red"),
                    name="nearest neighbors"), row=1, col=1)
        
        temp1 = pd.DataFrame(embedding_smiles1.loc[[molecule],:])
        temp2 = pd.DataFrame(embedding_smiles2.loc[[molecule],:])
        fig1.add_trace(go.Scattergl(x=[float(temp1.loc[molecule,'x'])],y=[float(temp1.loc[molecule,'y'])],mode="markers",
                    marker=dict(size=5, color="Red",line=dict(color='Black',width=1)),
                    name="clicked"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=[float(temp2.loc[molecule,'x'])],y=[float(temp2.loc[molecule,'y'])],mode="markers",
                    marker=dict(size=5, color="Red",line=dict(color='Black',width=1)),
                    name="clicked"), row=1, col=1)

    return fig1, fig2, None, None

@callback(
    Output('model1-umap', 'figure',allow_duplicate=True),
    Output('model2-umap', 'figure',allow_duplicate=True),
    Input('model1-umap','selectedData'),
    Input('model2-umap','selectedData'),
    State('model1-umap','clickData'),
    State('model2-umap','clickData'),
    State('dropdown-model1', 'value'),
    State('dropdown-model2', 'value'),
    State('dropdown-color','value'),
    prevent_initial_call=True,
)
def update_selected_data(selectedData1, selectedData2, clickData1, clickData2, value1, value2, color_property):
    input_id = ctx.triggered_id

    global model1, model2, embedding_smiles1, embedding_smiles2, molecule_name_index, clicked_data, selected_data, nearest_nbrs, molecules


    fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings",color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline')
    fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings",color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline')

    if (color_property == 'None') or ((color_property in property_truths_names) and (len(property_truths_name_file[color_property].index.intersection(model_similarities[value1].index))==0)):
            fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
            fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")

    
    if clickData1 is not None or clickData2 is not None:
        data_row = clicked_data.squeeze()

        molecule = str(data_row['SMILES'])
        
        if color_property == 'similarity':
            embedding_smiles1.iloc[:,-1] = model1_similarity.loc[:,molecule]
            embedding_smiles2.iloc[:,-1] = model2_similarity.loc[:,molecule]

        fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings",color=embedding_smiles1.iloc[:,-1].name,color_continuous_scale='haline')
        fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings",color=embedding_smiles2.iloc[:,-1].name,color_continuous_scale='haline')

        if (color_property == 'None') or ((color_property in property_truths_names) and (len(property_truths_name_file[color_property].index.intersection(model_similarities[value1].index))==0)):
            fig1 = px.scatter(embedding_smiles1,x='x',y='y',title="UMAP of " + value1 + " Embeddings")
            fig2 = px.scatter(embedding_smiles2,x='x',y='y',title="UMAP of " + value2 + " Embeddings")

        molecules1 = nearest_nbrs.iloc[molecule_name_index[molecule],[x + 10*model_name_index[value1] for x in list(range(10))]]
        molecules2 = nearest_nbrs.iloc[molecule_name_index[molecule],[x + 10*model_name_index[value2] for x in list(range(10))]]
        mol_smiles1 = [molecule_index_name[molecules1[x]] for x in range(len(molecules1))]
        mol_smiles2 = [molecule_index_name[molecules2[x]] for x in range(len(molecules2))]
        fig1.add_trace(go.Scattergl(x=list(embedding_smiles1.loc[mol_smiles1,'x'].astype(float)),y=list(embedding_smiles1.loc[mol_smiles1,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="Red"),
                    name="nearest neighbors"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=list(embedding_smiles2.loc[mol_smiles2,'x'].astype(float)),y=list(embedding_smiles2.loc[mol_smiles2,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="Red"),
                    name="nearest neighbors"), row=1, col=1)
        
        temp1 = pd.DataFrame(embedding_smiles1.loc[[molecule],:])
        temp2 = pd.DataFrame(embedding_smiles2.loc[[molecule],:])
        fig1.add_trace(go.Scattergl(x=[float(temp1.loc[molecule,'x'])],y=[float(temp1.loc[molecule,'y'])],mode="markers",
                    marker=dict(size=5, color="Red",line=dict(color='Black',width=1)),
                    name="clicked"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=[float(temp2.loc[molecule,'x'])],y=[float(temp2.loc[molecule,'y'])],mode="markers",
                    marker=dict(size=5, color="Red",line=dict(color='Black',width=1)),
                    name="clicked"), row=1, col=1)
        

    
    if ((selectedData1 is not None) and (len(selectedData1) != 0)) or ((selectedData2 is not None) and (len(selectedData2) != 0)):
        if input_id == 'model1-umap' and selectedData1 is not None:
            pts = selectedData1["points"][:]
            print(pts)
            if len(pts) != 0:
                pointIndexes = []
                for x in range(len(pts)):
                    if pts[x]['curveNumber'] == 0:
                        pointIndexes.append(pts[x]['pointIndex'])
                selected_data = embedding_smiles1.reset_index().loc[pointIndexes]
        elif input_id == 'model2-umap' and selectedData2 is not None:
            pts = selectedData2["points"][:]
            if len(pts) != 0:
                pointIndexes = []
                for x in range(len(pts)):
                    if pts[x]['curveNumber'] == 0:
                        pointIndexes.append(pts[x]['pointIndex'])
                selected_data = embedding_smiles2.reset_index().loc[pointIndexes]


        selected_molecules = list(selected_data['SMILES'])
        temp1 = pd.DataFrame(embedding_smiles1.loc[selected_molecules,:])
        temp2 = pd.DataFrame(embedding_smiles2.loc[selected_molecules,:])

        fig1.add_trace(go.Scattergl(x=list(temp1.loc[selected_molecules,'x'].astype(float)),y=list(temp1.loc[selected_molecules,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="magenta"),
                    name="selected"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=list(temp2.loc[selected_molecules,'x'].astype(float)),y=list(temp2.loc[selected_molecules,'y'].astype(float)),mode="markers",
                    marker=dict(size=5, color="magenta"),
                    name="selected"), row=1, col=1)
        
        # fig1.update_layout(legend_traceorder="reversed")
        # fig2.update_layout(legend_traceorder="reversed")

    return fig1, fig2

@callback(
    Output('kneighbor-map', 'figure'),
    Input('num-nbrs','value'),
)
def update_heatmap(num_nbrs):
    global model_overlaps_melted
    heatmap = model_overlaps_melted.copy()
    heatmap = heatmap[heatmap['num_nbrs']==num_nbrs].iloc[:,1:].pivot(index='model1',columns='model2').droplevel(0,axis=1)
    heatmap.insert(len(model_names)-1,str(len(model_names)-1),[0]*len(heatmap))
    heatmap = heatmap.transpose()
    heatmap.insert(0,'0.0',[0]*len(heatmap))
    heatmap = heatmap.transpose()
    for i in range(len(heatmap)):
        heatmap.iloc[i,i] = np.nan
        for j in range(1,len(heatmap.iloc[:,0])):
            heatmap.iloc[i,j] = heatmap.iloc[j,i]

    heatmap.index = model_names
    heatmap.columns = model_names

    fig = px.imshow(heatmap,title='Pairwise % Overlap in Nearest ' + str(num_nbrs) + ' Neighbors Between All Models')

    return fig

@callback(
    Output('property-similarity-scatter1', 'figure'),
    Output('property-similarity-scatter2', 'figure'),
    Input('dropdown-scatter-model1','value'),
    Input('dropdown-scatter-model2','value'),
    Input('dropdown-feature','value'),
)
def update_scatters(value1,value2,feature):
    global model_similarities,molecule_index_name, pairwise_data1, pairwise_data2

    if feature in property_truths_names:
        intersected_molecules = property_truths_name_file[feature].index.intersection(model_similarities[value1].index)
        num_molecules = min(len(intersected_molecules),400)
        output_dict = dict(sample([list(molecule_index_name.copy().items())[molecule_name_index[x]] for x in intersected_molecules],num_molecules))

        model1 = model_similarities[value1].copy().loc[list(output_dict.values()),list(output_dict.values())]
        model1.columns = list(output_dict.keys())
        model1 = model1.melt().rename(columns={'value':'cosine similarity distance'})
        model1.insert(1,'variable2',model1.index % num_molecules)
        model1['variable2'] = list(map(lambda x: list(output_dict.keys())[x],model1['variable2']))
        
        model1['variable'] = [molecule_index_name[x] for x in model1['variable']]
        model1['variable2'] = [molecule_index_name[x] for x in model1['variable2']]
        model1['Δ' + feature] = abs(pd.DataFrame(property_truths_name_file[feature].loc[model1['variable'],property_truths_name_file[feature].columns[-1]]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,property_truths_name_file[feature].columns[-1]] - pd.DataFrame(property_truths_name_file[feature].loc[model1['variable2'],property_truths_name_file[feature].columns[-1]].reset_index()).apply(pd.to_numeric,errors='coerce').loc[:,property_truths_name_file[feature].columns[-1]])

        model1['variable'] = [molecule_name_index[x] for x in model1['variable']]
        model1['variable2'] = [molecule_name_index[x] for x in model1['variable2']]

        model2 = model_similarities[value1].copy().loc[list(output_dict.values()),list(output_dict.values())]
        model2.columns = list(output_dict.keys())
        model2 = model2.melt().rename(columns={'value':'cosine similarity distance'})
        model2.insert(1,'variable2',model2.index % num_molecules)
        model2['variable2'] = list(map(lambda x: list(output_dict.keys())[x],model2['variable2']))
        
        model2['variable'] = [molecule_index_name[x] for x in model2['variable']]
        model2['variable2'] = [molecule_index_name[x] for x in model2['variable2']]
        model2['Δ' + feature] = abs(pd.DataFrame(property_truths_name_file[feature].loc[model2['variable'],property_truths_name_file[feature].columns[-1]]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,property_truths_name_file[feature].columns[-1]] - pd.DataFrame(property_truths_name_file[feature].loc[model2['variable2'],property_truths_name_file[feature].columns[-1]].reset_index()).apply(pd.to_numeric,errors='coerce').loc[:,property_truths_name_file[feature].columns[-1]])

        model2['variable'] = [molecule_name_index[x] for x in model2['variable']]
        model2['variable2'] = [molecule_name_index[x] for x in model2['variable2']]
    else:
        sample_size = min(min(max(int(len(all_molecules)/10),200),len(all_molecules)),400)
        output_dict = dict(sample(list(molecule_index_name.copy().items()), sample_size))

        model1 = model_similarities[value1].copy().loc[list(output_dict.values()),list(output_dict.values())]
        model1.columns = list(output_dict.keys())
        model1 = model1.melt().rename(columns={'value':'cosine similarity distance'})
        model1.insert(1,'variable2',model1.index % sample_size)
        model1['variable2'] = list(map(lambda x: list(output_dict.keys())[x],model1['variable2']))

        model1['Δ' + feature] = abs(pd.DataFrame(mol_features.loc[model1['variable'],feature]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,feature] - pd.DataFrame(mol_features.loc[model1['variable2'],feature]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,feature])
        
        model2 = model_similarities[value2].copy().loc[list(output_dict.values()),list(output_dict.values())]
        model2.columns = list(output_dict.keys())
        model2 = model2.melt().rename(columns={'value':'cosine similarity distance'})
        model2.insert(1,'variable2',model2.index % sample_size)
        model2['variable2'] = list(map(lambda x: list(output_dict.keys())[x],model2['variable2']))

        model2['Δ' + feature] = abs(pd.DataFrame(mol_features.loc[model2['variable'],feature]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,feature] - pd.DataFrame(mol_features.loc[model2['variable2'],feature]).reset_index().apply(pd.to_numeric,errors='coerce').loc[:,feature])

        
    pairwise_data1 = model1.copy()

    correlation = model1.iloc[:,2:].corr('pearson').iloc[0,1]

    fig1 = px.scatter(model1,x='cosine similarity distance',y='Δ' + feature,title='Cosine Similarity vs Difference in ' + feature + ' for Each Pair of a <br>Random Sample of Molecules in ' + value1 + '<br>' + 'Pearson correlation: ' + str(correlation))
    fig1.update_traces(marker=dict(size=5,opacity=0.1),
                    selector=dict(mode='markers'))

    
    pairwise_data2 = model2.copy()

    correlation = model2.iloc[:,2:].corr('pearson').iloc[0,1]

    fig2 = px.scatter(model2,x='cosine similarity distance',y='Δ' + feature,title='Cosine Similarity vs Difference in ' + feature + ' for Each Pair of a <br>Random Sample of Molecules in ' + value2 + '<br>' + 'Pearson correlation: ' + str(correlation))
    fig2.update_traces(marker=dict(size=5,opacity=0.1),
                    selector=dict(mode='markers'))

    return fig1, fig2

@callback(
    Output('property-similarity-scatter1', 'figure',allow_duplicate=True),
    Output('property-similarity-scatter2', 'figure',allow_duplicate=True),
    Input('property-similarity-scatter1','clickData'),
    Input('property-similarity-scatter2','clickData'),
    State('dropdown-model1', 'value'),
    State('dropdown-model2', 'value'),
    State('dropdown-feature','value'),
    prevent_initial_call=True,
)
def update_clicked_data(clickData1, clickData2, value1, value2, feature):
    input_id = ctx.triggered_id

    global molecule_name_index, pairwise_data1, pairwise_data2

    pairwise_data1['clicked'] = 0
    pairwise_data2['clicked'] = 0

    if clickData1 is not None or clickData2 is not None:
        if input_id == 'property-similarity-scatter1':
            pt = clickData1["points"][0]['pointIndex']
            
            clicked_data = pairwise_data1.loc[pt]
            data_row = clicked_data.squeeze()
        elif input_id == 'property-similarity-scatter2':
            pt = clickData2["points"][0]['pointIndex']
            
            clicked_data = pairwise_data2.loc[pt]
            data_row = clicked_data.squeeze()

        molecules = [data_row['variable'],data_row['variable2']]
        pairwise_data1.loc[pairwise_data1['variable']==molecules[0],'clicked'] = 1
        pairwise_data1.loc[pairwise_data1['variable']==molecules[1],'clicked'] = -1
        pairwise_data1.loc[pairwise_data1['variable2']==molecules[0],'clicked'] = 1
        pairwise_data1.loc[pairwise_data1['variable2']==molecules[1],'clicked'] = -1
        pairwise_data2.loc[pairwise_data2['variable']==molecules[0],'clicked'] = 1
        pairwise_data2.loc[pairwise_data2['variable']==molecules[1],'clicked'] = -1
        pairwise_data2.loc[pairwise_data2['variable2']==molecules[0],'clicked'] = 1
        pairwise_data2.loc[pairwise_data2['variable2']==molecules[1],'clicked'] = -1

        pairwise_data1.loc[(pairwise_data1['variable']==molecules[0]) & (pairwise_data1['variable2']==molecules[1]),'clicked'] = 2
        pairwise_data1.loc[(pairwise_data1['variable']==molecules[1]) & (pairwise_data1['variable2']==molecules[0]),'clicked'] = 2
        pairwise_data2.loc[(pairwise_data2['variable']==molecules[0]) & (pairwise_data2['variable2']==molecules[1]),'clicked'] = 2
        pairwise_data2.loc[(pairwise_data2['variable']==molecules[1]) & (pairwise_data2['variable2']==molecules[0]),'clicked'] = 2

        
        model1_mol1 = pairwise_data1.loc[pairwise_data1['clicked']==1]
        model1_mol2 = pairwise_data1.loc[pairwise_data1['clicked']==-1]
        model2_mol1 = pairwise_data2.loc[pairwise_data2['clicked']==1]
        model2_mol2 = pairwise_data2.loc[pairwise_data2['clicked']==-1]
        clicked1 = pairwise_data1.loc[pairwise_data1['clicked']==2]
        clicked2 = pairwise_data2.loc[pairwise_data2['clicked']==2]

        correlation = pairwise_data1.iloc[:,2:4].corr('pearson').iloc[0,1]

        fig1 = px.scatter(pairwise_data1,x='cosine similarity distance',y='Δ' + feature,title='Cosine Similarity vs Difference in ' + feature + ' for Each Molecule Pair in ' + value1 + '<br>' + 'Pearson correlation: ' + str(correlation), height=700)
        fig1.update_traces(marker=dict(size=5,opacity=0.1),
                        selector=dict(mode='markers'))
        fig1.add_trace(go.Scattergl(x=list(model1_mol1.astype(float)['cosine similarity distance']),y=list(model1_mol1.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=4, color="yellow"),
                    name="molecule 1"))
        fig1.add_trace(go.Scattergl(x=list(model1_mol2.astype(float)['cosine similarity distance']),y=list(model1_mol2.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=4, color="blue"),
                    name="molecule 2"))
        fig1.add_trace(go.Scattergl(x=list(clicked1.astype(float)['cosine similarity distance']),y=list(clicked1.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=5, color="red"),
                    name="clicked"))

        correlation = pairwise_data2.iloc[:,2:4].corr('pearson').iloc[0,1]

        fig2 = px.scatter(pairwise_data2,x='cosine similarity distance',y='Δ' + feature,title='Cosine Similarity vs Difference in ' + feature + ' for Each Molecule Pair in ' + value2 + '<br>' + 'Pearson correlation: ' + str(correlation), height=700)
        fig2.update_traces(marker=dict(size=5,opacity=0.1),
                        selector=dict(mode='markers'))
        fig2.add_trace(go.Scattergl(x=list(model2_mol1.astype(float)['cosine similarity distance']),y=list(model2_mol1.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=4, color="yellow"),
                    name="molecule 1"))
        fig2.add_trace(go.Scattergl(x=list(model2_mol2.astype(float)['cosine similarity distance']),y=list(model2_mol2.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=4, color="blue"),
                    name="molecule 2"))
        fig2.add_trace(go.Scattergl(x=list(clicked2.astype(float)['cosine similarity distance']),y=list(clicked2.astype(float)['Δ' + feature]),mode="markers",
                    marker=dict(size=5, color="red"),
                    name="clicked"))
        
    else:
        return no_update, no_update


    return fig1, fig2


@callback(
    Output('property-correlation-scatter', 'figure'),
    Output('property-prediction-scatter', 'figure'),
    Input('dropdown-corr-pred-scatter-model1','value'),
    Input('dropdown-corr-pred-scatter-model2','value'),
)
def update_scatters(value1,value2):
    global property_correlations, property_predictions

    corrs = property_correlations.copy().transpose()
    scores = property_predictions.copy().transpose()

    fig1 = px.scatter(corrs,x=value1,y=value2,hover_name=corrs.index,color=itemgetter(*list(corrs.index))(property_difference_index),
                      title='Scatter of Correlations Between Cosine Similarity Distance and <br>Differences in Different Molecular Property Descriptors for Two Models'
                      ).update_layout(coloraxis_colorbar=dict(title="property_#"))
    fig1.add_trace(go.Scattergl(x=[min(min(corrs[value1]),min(corrs[value2])),max(max(corrs[value1]),max(corrs[value2]))],y=[min(min(corrs[value1]),min(corrs[value2])),max(max(corrs[value1]),max(corrs[value2]))],mode='lines'))

    fig2 = px.scatter(scores,x=value1,y=value2,hover_name=corrs.index,color=itemgetter(*list(scores.index))(property_index),
                      title='Scatter of R^2 Scores of ML Embedding Predictions on <br>Different Molecular Property Descriptors for Two Models'
                      ).update_layout(coloraxis_colorbar=dict(title="property_#"))
    fig2.add_trace(go.Scattergl(x=[min(min(scores[value1]),min(scores[value2])),max(max(scores[value1]),max(scores[value2]))],y=[min(min(scores[value1]),min(scores[value2])),max(max(scores[value1]),max(scores[value2]))],mode='lines'))

    return fig1, fig2

@callback(
    Output('pairwise-property-correlation-heatmap', 'figure'),
    Output('pairwise-property-prediction-heatmap', 'figure'),
    Output('prediction-vs-correlation-scatter-properties', 'figure'),
    Output('prediction-vs-correlation-scatter-models', 'figure'),
    Output('prediction-vs-correlation-buckets', 'figure'),
    Output('prediction-vs-correlation-scatter-bar', 'figure'),
    Input('data','data'),
    State('input-good-corr','value'),
    State('input-bad-corr','value'),
    State('input-good-pred','value'),
    State('input-bad-pred','value'),
)
def update_correlation_and_prediction_scatters(data,good_corr,bad_corr,good_pred,bad_pred):
    global property_correlations, property_predictions

    corrs = property_correlations.copy()
    scores = property_predictions.copy()
    corrs.columns = scores.columns

    temp = property_correlations.iloc[:,:].transpose()
    temp_corr = temp.corr()

    corr_fig_heatmap = px.imshow(temp_corr,title="Correlations of Correlations Between Cosine Similarities and Molecular Property Descriptor Differences"
                                 ).update_layout(xaxis_title='',yaxis_title='')

    temp = property_predictions.iloc[:,:].transpose()
    temp_corr = temp.corr()

    pred_fig_heatmap = px.imshow(temp_corr,title="Correlations of R^2 Scores of ML Predictions on Molecular Property Descriptors from Embeddings"
                                 ).update_layout(xaxis_title='',yaxis_title='')

    c = corrs.melt().rename(columns={"Unnamed: 0":"property","value":"corrs"})
    c['model'] = [model_index_name[c.index[x] % len(model_names)] for x in range(len(c.index))]
    s = scores.melt().rename(columns={"Unnamed: 0":"property","value":"scores"})
    s['model'] = [model_index_name[s.index[x] % len(model_names)] for x in range(len(s.index))]

    fig_properties = px.scatter(x=c.corrs,y=s.scores,color=c['property'],hover_name=c['model'],
                                title="Pearson Correlation vs R2_score Per Model Per Molecular Property Descriptor")
    fig_properties.update_layout(xaxis_title="Pearson Correlation",yaxis_title="R^2 Score")

    fig_models = px.scatter(x=c.corrs,y=s.scores,color=c['model'],hover_name=c['property'],
                            title="Pearson Correlation vs R2_score Per Model Per Molecular Property Descriptor")
    fig_models.update_layout(xaxis_title="Pearson Correlation",yaxis_title="R^2 Score")

    fig_bar = px.bar(corrs.corrwith(scores,axis=1))
    fig_bar.update_layout(showlegend=False,xaxis_title="model",yaxis_title="Pearson correlation",
                          title="Correlation Between Property Correlation and ML Prediction R^2 Score")
    
    merged = c.merge(s,on=["property","model"])

    good_corr = float(good_corr)
    bad_corr = float(bad_corr)
    good_pred = float(good_pred)
    bad_pred = float(bad_pred)

    buckets = pd.DataFrame(columns=["good corr, good pred","bad corr, good pred","good corr, bad pred","bad corr, bad pred"],index=model_names)
    buckets = buckets.rename_axis(columns="# of properties for")
    buckets.loc[:,"good corr, good pred"] = merged[(merged['corrs']<good_corr)&(merged['scores']>good_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"bad corr, good pred"] = merged[(merged['corrs']>bad_corr)&(merged['scores']>good_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"good corr, bad pred"] = merged[(merged['corrs']<good_corr)&(merged['scores']<bad_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"bad corr, bad pred"] = merged[(merged['corrs']>bad_corr)&(merged['scores']<bad_pred)].groupby('model').count().iloc[:,0]
    buckets = buckets.fillna(0).astype(int)

    fig_buckets = px.bar(buckets).update_layout(xaxis_title="model",yaxis_title="# of properties",title="Number of Properties per Threshold Bucket By Model")


    return corr_fig_heatmap, pred_fig_heatmap, fig_properties, fig_models, fig_buckets, fig_bar


@callback(
    Output('prediction-vs-correlation-buckets', 'figure', allow_duplicate=True),
    Input('input-good-corr','value'),
    Input('input-bad-corr','value'),
    Input('input-good-pred','value'),
    Input('input-bad-pred','value'),
    prevent_initial_call=True,
)
def update_bucket_fig(good_corr,bad_corr,good_pred,bad_pred):
    global property_correlations, property_predictions

    corrs = property_correlations.copy()
    scores = property_predictions.copy()
    corrs.columns = scores.columns

    c = corrs.melt().rename(columns={"Unnamed: 0":"property","value":"corrs"})
    c['model'] = [model_index_name[c.index[x] % len(model_names)] for x in range(len(c.index))]
    s = scores.melt().rename(columns={"Unnamed: 0":"property","value":"scores"})
    s['model'] = [model_index_name[s.index[x] % len(model_names)] for x in range(len(s.index))]

    merged = c.merge(s,on=["property","model"])

    good_corr = float(good_corr)
    bad_corr = float(bad_corr)
    good_pred = float(good_pred)
    bad_pred = float(bad_pred)

    buckets = pd.DataFrame(columns=["good corr, good pred","good corr, bad pred","bad corr, good pred","bad corr, bad pred"],index=model_names)
    buckets = buckets.rename_axis(columns="# of properties for")
    buckets.loc[:,"good corr, good pred"] = merged[(merged['corrs']<good_corr)&(merged['scores']>good_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"bad corr, good pred"] = merged[(merged['corrs']>bad_corr)&(merged['scores']>good_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"good corr, bad pred"] = merged[(merged['corrs']<good_corr)&(merged['scores']<bad_pred)].groupby('model').count().iloc[:,0]
    buckets.loc[:,"bad corr, bad pred"] = merged[(merged['corrs']>bad_corr)&(merged['scores']<bad_pred)].groupby('model').count().iloc[:,0]
    buckets = buckets.fillna(0).astype(int)

    fig_buckets = px.bar(buckets)


    return fig_buckets

@callback(
    Output('threshold-data', 'data'),
    Input('flow-input-good-corr','value'),
    Input('flow-input-bad-corr','value'),
    Input('flow-input-good-pred','value'),
    Input('flow-input-bad-pred','value'),
)
def update_bucket_thresholds(good_corr,bad_corr,good_pred,bad_pred):
    global corr_pred
    good_corr = float(good_corr)
    bad_corr = float(bad_corr)
    good_pred = float(good_pred)
    bad_pred = float(bad_pred)

    for row in corr_pred.index:
        if(corr_pred.loc[row,'corrs']<good_corr)&(corr_pred.loc[row,'scores']>good_pred):
            corr_pred.loc[row,'bucket_model'] = 'good corr, good pred: ' + corr_pred.loc[row,'model']
            corr_pred.loc[row,'bucket'] = 'good corr, good pred'
        elif(corr_pred.loc[row,'corrs']<good_corr)&(corr_pred.loc[row,'scores']<bad_pred):
            corr_pred.loc[row,'bucket_model'] = 'good corr, bad pred: ' + corr_pred.loc[row,'model']
            corr_pred.loc[row,'bucket'] = 'good corr, bad pred'
        elif(corr_pred.loc[row,'corrs']>bad_corr)&(corr_pred.loc[row,'scores']>good_pred):
            corr_pred.loc[row,'bucket_model'] = 'bad corr, good pred: ' + corr_pred.loc[row,'model']
            corr_pred.loc[row,'bucket'] = 'bad corr, good pred'
        elif(corr_pred.loc[row,'corrs']>bad_corr)&(corr_pred.loc[row,'scores']<bad_pred):
            corr_pred.loc[row,'bucket_model'] = 'bad corr, bad pred: ' + corr_pred.loc[row,'model']
            corr_pred.loc[row,'bucket'] = 'bad corr, bad pred'

    return pd.DataFrame().to_json(orient="split")

@callback(
    Output('property-corr-pred-scatter', 'figure'),
    Output('property-corr-pred-sankey', 'figure'),
    Input('dropdown-flow-model1','value'),
    Input('dropdown-flow-model2','value'),
    Input('dropdown-start','value'),
    Input('dropdown-end','value'),
    Input('threshold-data','data')
)
def update_property_flow_graphs(value1,value2,start_bucket,end_bucket,data):
    input_id = ctx.triggered_id
    global property_correlations, property_predictions, model_relationships, corr_pred

    if value1 == 'None' and value2 == 'None':
        return no_update, no_update

    if value1 != 'None' and value2 != 'None':
        #compare two models
        model1 = corr_pred[corr_pred['model']==value1].set_index('property')
        model2 = corr_pred[corr_pred['model']==value2].set_index('property')

        temp1 = model1[model1['bucket_model']!=model2['bucket_model']]
        temp2 = model2[model1['bucket_model']!=model2['bucket_model']]
        link_values1 = Counter([(temp1.loc[x,'bucket_model'],temp2.loc[x,'bucket_model']) for x in temp1.index])

        labels = list(set([list(link_values1.keys())[x][0] for x in range(len(link_values1))]+[list(link_values1.keys())[x][1] for x in range(len(link_values1))]))
        labels_name_index = dict(zip(labels,list(range(len(labels)))))
        sources = [labels_name_index[list(link_values1.keys())[x][0]] for x in range(len(link_values1))]
        targets = [labels_name_index[list(link_values1.keys())[x][1]] for x in range(len(link_values1))]
        thicknesses = [list(link_values1.values())[x] for x in range(len(link_values1))]
        colors = []
        for label in labels:
            if 'good corr, good pred' in label:
                colors.append("blue")
            elif 'bad corr, good pred' in label:
                colors.append("green")
            elif 'bad corr, bad pred' in label:
                colors.append("yellow")
            elif 'good corr, bad pred' in label:
                colors.append("red")

        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = colors
            ),
            link = dict(
            source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = targets,
            value = thicknesses
        ))])

        fig_sankey.update_layout(title_text="Sankey Diagram of Property Descriptor Flow Between Threshold Buckets", font_size=10)


        scatter_data = corr_pred.set_index('model').loc[[value1, value2],:].reset_index()
        if start_bucket == 'Any':
            start1 = scatter_data[(scatter_data['model']==value1)]
        else:
            start1 = scatter_data[(scatter_data['model']==value1)&(scatter_data['bucket']==start_bucket)]
            
        if end_bucket == 'Any':
            end1 = scatter_data[(scatter_data['model']==value2)]
        else:
            end1 = scatter_data[(scatter_data['model']==value2)&(scatter_data['bucket']==end_bucket)]

        start1 = start1.set_index('property').loc[np.intersect1d(start1['property'],end1['property']),:].reset_index()
        end1 = end1.set_index('property').loc[np.intersect1d(start1['property'],end1['property']),:].reset_index()
        

        fig_properties = px.scatter(x=scatter_data.corrs,y=scatter_data.scores,color=scatter_data['model'],hover_name=scatter_data['property'],
                                    title="Pearson Correlation vs R2_score Per Model Per Molecular Property Descriptor")
        fig_properties.update_layout(xaxis_title="Pearson Correlation",yaxis_title="R^2 Score")
        
        list_of_all_arrows = []
        for x0,y0,x1,y1 in zip(end1.corrs, end1.scores, start1.corrs, start1.scores):
            arrow = go.layout.Annotation(dict(
                            x=x0,
                            y=y0,
                            xref="x", yref="y",
                            text="",
                            showarrow=True,
                            axref="x", ayref='y',
                            ax=x1,
                            ay=y1,
                            arrowhead=5,
                            arrowwidth=1,
                            arrowcolor='rgb(0,0,0)',)
                        )
            list_of_all_arrows.append(arrow)
        fig_properties.update_layout(annotations=list_of_all_arrows)

    elif model_relationships is not None:
        #Esmi specific training process
        if value1 == 'None':
            model = value2
        elif value2 == 'None':
            model = value1

        model_set = []
        temp = model
        while temp in list(model_relationships['child']):
            model_set.append(temp)
            temp = model_relationships.loc[model_relationships['child']==temp,'parent'].item()

        model_set.append(temp)

        model_set.reverse()
    
        models = [corr_pred[corr_pred['model']==model_set[x]].set_index('property') for x in range(len(model_set))]

        link_values = [Counter([(models[y].loc[x,'bucket_model'],models[y+1].loc[x,'bucket_model']) for x in models[y].index]) for y in range(len(models)-1)]

        labels = []
        for y in range(len(link_values)):
            labels = labels + list(set([list(link_values[y].keys())[x][0] for x in range(len(link_values[y]))]+[list(link_values[y].keys())[x][1] for x in range(len(link_values[y]))]))
        labels_name_indexes = dict(zip(labels,list(range(len(labels)))))
        sources = []
        for y in range(len(link_values)):
            sources = sources + [labels_name_indexes[list(link_values[y].keys())[x][0]] for x in range(len(link_values[y]))]
        targets = []
        for y in range(len(link_values)):
            targets = targets + [labels_name_indexes[list(link_values[y].keys())[x][1]] for x in range(len(link_values[y]))]
        thicknesses = []
        for y in range(len(link_values)):
            thicknesses = thicknesses + [list(link_values[y].values())[x] for x in range(len(link_values[y]))]
        colors = []
        for label in labels:
            if 'good corr, good pred' in label:
                colors.append("blue")
            elif 'bad corr, good pred' in label:
                colors.append("green")
            elif 'bad corr, bad pred' in label:
                colors.append("yellow")
            elif 'good corr, bad pred' in label:
                colors.append("red")

        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = colors
            ),
            link = dict(
            source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = targets,
            value = thicknesses
        ))])

        fig_sankey.update_layout(title_text="Sankey Diagram of Property Descriptor Flow Between Threshold Buckets", font_size=10)

        scatter_data = corr_pred.set_index('model').loc[model_set,:].reset_index()
        if start_bucket == 'Any':
            starts = [corr_pred[(corr_pred['model']==model_set[x])] for x in range(len(model_set)-1)]
        else:
            starts = [corr_pred[(corr_pred['model']==model_set[x])&(corr_pred['bucket']==start_bucket)] for x in range(len(model_set)-1)]
            
        if end_bucket == 'Any':
            ends = [corr_pred[(corr_pred['model']==model_set[x])] for x in range(1,len(model_set))]
        else:
            ends = [corr_pred[(corr_pred['model']==model_set[x])&(corr_pred['bucket']==end_bucket)] for x in range(1,len(model_set))]

        starts = [starts[x].set_index('property').loc[np.intersect1d(starts[x]['property'],ends[x]['property']),:].reset_index() for x in range(len(model_set)-1)]
        ends = [ends[x].set_index('property').loc[np.intersect1d(starts[x]['property'],ends[x]['property']),:].reset_index() for x in range(len(model_set)-1)]
        

        fig_properties = px.scatter(x=scatter_data.corrs,y=scatter_data.scores,color=scatter_data['model'],hover_name=scatter_data['property'],
                                    title="Pearson Correlation vs R2_score Per Model Per Molecular Property Descriptor")
        fig_properties.update_layout(xaxis_title="Pearson Correlation",yaxis_title="R^2 Score")
        
        list_of_all_arrows = []
        for i in range(len(model_set)-1):
            for x0,y0,x1,y1 in zip(ends[i].corrs, ends[i].scores, starts[i].corrs, starts[i].scores):
                arrow = go.layout.Annotation(dict(
                                x=x0,
                                y=y0,
                                xref="x", yref="y",
                                text="",
                                showarrow=True,
                                axref="x", ayref='y',
                                ax=x1,
                                ay=y1,
                                arrowhead=5,
                                arrowwidth=1,
                                arrowcolor='rgb(0,0,0)',)
                            )
                list_of_all_arrows.append(arrow)
        fig_properties.update_layout(annotations=list_of_all_arrows)
    else:
        return no_update, no_update
        

    return fig_properties, fig_sankey


@callback(
    Output('network-graph-spearman', 'figure'),
    Input('input-spearman','value'),
    Input('dropdown-operator-spearman','value'),
)
def update_network_spearman(spearman,operator):
    global pairwise_spearmanr
    pairwise_spearmanr_avg = pairwise_spearmanr.iloc[:,:2]
    avg = pairwise_spearmanr.iloc[:,2:].mean(axis=1)
    pairwise_spearmanr_avg.insert(2,"average",pd.DataFrame(avg))

    pairwise_spearmanr_avg.loc[:,'model1'] = [model_index_name[x] for x in pairwise_spearmanr_avg['model1']]
    pairwise_spearmanr_avg.loc[:,'model2'] = [model_index_name[x] for x in pairwise_spearmanr_avg['model2']]
    
    if operator == '<':
        new = pairwise_spearmanr_avg[(pairwise_spearmanr_avg['average']<float(spearman))]
    elif operator == '>':
        new = pairwise_spearmanr_avg[(pairwise_spearmanr_avg['average']<1) & (pairwise_spearmanr_avg['average']>float(spearman))]
    edge_labels = dict(zip(tuple(zip(list(new.iloc[:,0].values),list(new.iloc[:,1].values))),list(round(x,2) for x in new.iloc[:,2].values)))

    G = nx.from_pandas_edgelist(new,"model1","model2",True)
    pos = nx.spring_layout(G)

    edge_x = [x for n0, n1 in G.edges for x in (pos[n0][0], pos[n1][0], None)]
    edge_y = [y for n0, n1 in G.edges for y in (-pos[n0][1], -pos[n1][1], None)]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [x for x, y in pos.values()]
    node_y = [-y for x, y in pos.values()]
    node_name = [key for key in pos.keys()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertemplate =
        '<b>%{text}</b>',
        text = ['Name: {}'.format(x) for x in node_name],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    i = 0
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_name[i] +': '+str(len(adjacencies[1])) + ' connections')
        i += 1

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Edges weighted by Spearman correlation between models',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',)
                    )

    return fig

@callback(
    Output('network-graph-kneighbor', 'figure'),
    Input('input-kneighbor','value'),
    Input('dropdown-operator-kneighbor','value'),
)
def update_network_kneighbor(overlap,operator):
    global model_overlaps_melted
    new = model_overlaps_melted[model_overlaps_melted['num_nbrs']==model_overlaps.index[int(statistics.median(range(len(model_overlaps.index))))]]

    new.loc[:,'model1'] = [model_index_name[x] for x in new['model1']]
    new.loc[:,'model2'] = [model_index_name[x] for x in new['model2']]

    if operator == '<':
        new = new[(new['overlap_percent']<float(overlap))]
    elif operator == '>':
        new = new[(new['overlap_percent']<1) & (new['overlap_percent']>float(overlap))]

    G = nx.from_pandas_edgelist(new,"model1","model2",True)
    pos = nx.spring_layout(G)

    edge_x = [x for n0, n1 in G.edges for x in (pos[n0][0], pos[n1][0], None)]
    edge_y = [y for n0, n1 in G.edges for y in (-pos[n0][1], -pos[n1][1], None)]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [x for x, y in pos.values()]
    node_y = [-y for x, y in pos.values()]
    node_name = [key for key in pos.keys()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertemplate =
        '<b>%{text}</b>',
        text = ['Name: {}'.format(x) for x in node_name],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    i = 0
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_name[i] +': '+str(len(adjacencies[1])) + ' connections')
        i += 1

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Edges weighted by nearest ' + str(model_overlaps.index[int(statistics.median(range(len(model_overlaps.index))))]) + ' neighbors overlap between models',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',)
                    )

    return fig

@callback(
    Output('network-graph-correlation', 'figure'),
    Input('input-correlation','value'),
    Input('dropdown-operator-correlation','value'),
)
def update_network_correlation(correlation,operator):
    global property_correlations,property_predictions
    corrs = property_correlations.copy()
    scores = property_predictions.copy()
    corrs.columns = scores.columns

    new = corrs.transpose().corr().melt().rename(columns={"Unnamed: 0":"model1"})
    new.insert(1,'model2',new.index % len(model_names))
    new['model2'] = [model_index_name[x] for x in new['model2']]

    if operator == '<':
        new = new[(new['value']<float(correlation))]
    elif operator == '>':
        new = new[(new['value']<1) & (new['value']>float(correlation))]

    G = nx.from_pandas_edgelist(new,"model1","model2",True)
    pos = nx.spring_layout(G)

    edge_x = [x for n0, n1 in G.edges for x in (pos[n0][0], pos[n1][0], None)]
    edge_y = [y for n0, n1 in G.edges for y in (-pos[n0][1], -pos[n1][1], None)]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [x for x, y in pos.values()]
    node_y = [-y for x, y in pos.values()]
    node_name = [key for key in pos.keys()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertemplate =
        '<b>%{text}</b>',
        text = ['Name: {}'.format(x) for x in node_name],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    i = 0
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_name[i] +': '+str(len(adjacencies[1])) + ' connections')
        i += 1

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Edges weighted by correlation of correlations of differences in <br>cosine similarity and molecular properties between models',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',)
                    )

    return fig

@callback(
    Output('model-relationships-treemap', 'figure'),
    Input('data','data'),
)
def relationships_treemap(data):
    global model_relationships

    if model_relationships is not None:
        fig = px.treemap(names=list(model_relationships.child),parents=list(model_relationships.parent))

        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        
        return fig
    else:
        return no_update

if __name__ == '__main__':
    app.run(debug=True)
