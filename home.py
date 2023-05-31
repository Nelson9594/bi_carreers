import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import openpyxl
import numpy as np
from matplotlib.cm import get_cmap
import geopandas as gpd
import pydeck as pdk

# Nom de fichier local à charger si l'URL n'est pas disponible
filename = "/Users/nelson/Desktop/BI/data/Base_open_data_23.xlsx"

try:
    url = "https://www.data.gouv.fr/fr/datasets/r/cf654bbd-aa1e-458c-9d08-24252f66f16b"
    # Vérifier si l'URL est disponible
    urllib.request.urlopen(url)
    # Si l'URL est disponible, charger le contenu du fichier Excel
    data = pd.read_excel(url, sheet_name="BMO_2023_open_data")
    
except urllib.error.URLError:
    # Si l'URL n'est pas disponible, charger le contenu du fichier local
    data = pd.read_excel(filename, sheet_name="BMO_2023_open_data")


# Title
st.markdown(
    "<h1 style='text-align: center;'>Projet Business Intelligences sur le domaine des emplois en France</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center;'>Présentation des données</h2>",
    unsafe_allow_html=True
)


st.write(data)

#DATA CLEANING
st.header('Data cleaning')

st.markdown("La procédure de nettoyage de donnée vise à identifier et à corriger les erreurs, les valeurs aberrantes, les doublons et les données manquantes dans un jeu de données, afin de garantir la qualité et la fiabilité des données utilisées pour l'analyse.")

st.markdown('<h4>1. Renommer les colonnes</h4>', unsafe_allow_html= True)

data.rename(columns={
    'annee': 'Année',
    'Code métier BMO': 'Code Métier BMO',
    'Nom métier BMO': 'Nom Métier BMO',
    'Famille_met': 'Code Famille Métier',
    'Lbl_fam_met': 'Label Famille Métier',
    'BE23': 'Code bassin emploi',
    'NOMBE23': 'Nom du bassin emploi',
    'Dept': 'Numero de Département',
    'NomDept': 'Nom du Département',
    'REG': 'code région INSEE',
    'NOM_REG': 'Nom de la Région',
    'met': 'Nombre de projet de recrutement',
    'xmet': 'Nombre de projet de recrutement jugés difficile',
    'smet': 'Nombre de projet de recrutement saisonniers'
}, inplace=True)

st.write(data.head())

st.markdown('<h4>2. Remplacer les cellules vide ou contenant des anomalie</h4>', unsafe_allow_html= True)

data_replace = data.replace('*', np.nan)
st.write(data_replace.head())

st.markdown('<h4>3. Supprimer les cellules NaN ou None</h4>', unsafe_allow_html= True)

new_data = data_replace.dropna()
st.write(new_data.head())



#EXPLORATION DATASET
st.header('Exploration Dataset')

#Projet de recrutement

st.markdown('<h4>Projet de recrutement</h4>', unsafe_allow_html= True)

st.caption('Selectionner les colonnes pertinentes')
code = '''colonnes = ['Nombre de projet de recrutement', 'Nombre de projet de recrutement jugés difficile', 'Nombre de projet de recrutement saisonniers']'''
st.code(code, language='python')

colonnes = ['Nombre de projet de recrutement', 'Nombre de projet de recrutement jugés difficile', 'Nombre de projet de recrutement saisonniers']
new_data_num = new_data[colonnes].astype(int)

def sum_new_data_num(a):
    return new_data_num[a].sum()

nbr_pr = sum_new_data_num('Nombre de projet de recrutement')
nbr_prjd = sum_new_data_num('Nombre de projet de recrutement jugés difficile')
nbr_prs = sum_new_data_num('Nombre de projet de recrutement saisonniers')

st.text("")
st.text("")
st.text("")


col1, col2, col3 = st.columns(3)
col1.metric("Projet de recrutement", nbr_pr)
col2.metric("Jugés difficile", nbr_prjd)
col3.metric("Saisonnier", nbr_prs)


st.text("")
st.text("")
st.text("")


st.caption("L'Île-de-France est la région avec le plus grand nombre de projets de recrutement en France, suivie par Auvergne-Rhône-Alpes et Nouvelle-Aquitaine. Les régions Occitanie et Provence-Alpes-Côte d'Azur complètent le top 5.")

column_bmo = ['Nom de la Région', 'Nombre de projet de recrutement']
column_selected = new_data[column_bmo]
column_selected['Nombre de projet de recrutement'] = pd.to_numeric(column_selected['Nombre de projet de recrutement'], errors='coerce').astype(int)
somme_par_region = column_selected.groupby('Nom de la Région').sum()
somme_par_region = somme_par_region.sort_values(by='Nombre de projet de recrutement')
st.bar_chart(somme_par_region['Nombre de projet de recrutement'])

st.text("")
st.text("")
st.text("")


st.markdown("L'Île-de-France, Auvergne-Rhône-Alpes, et Nouvelle-Aquitaine sont les régions les plus actives en termes de projets de recrutement en France. Elles bénéficient d'une concentration d'entreprises et d'activités économiques importantes. La demande de recrutement est également significative en Occitanie et en Provence-Alpes-Côte d'Azur, avec des secteurs clés tels que l'aérospatial, la technologie et le tourisme.")

somme_par_region_asc = somme_par_region.sort_values(by='Nombre de projet de recrutement', ascending=False)
top_regions = somme_par_region_asc.head(5)

st.bar_chart(top_regions)

st.text("")
st.text("")
st.text("")


st.markdown("Ces données fournissent un aperçu des projets de recrutement à l'échelle nationale, en mettant en évidence les régions avec le plus grand nombre total de projets de recrutement, les projets de recrutement jugés difficiles et les projets de recrutement saisonniers. Cette information peut être utilisée pour évaluer les défis spécifiques liés au recrutement dans différentes régions de France.")
st.markdown("Par ailleurs, il est intéressant de noter qu'il existe des difficultés de recrutement significatives dans les grandes régions de France. Cela indique que trouver des candidats qualifiés et compétents pour ces postes est un défi majeur. Les raisons de ces difficultés peuvent varier, notamment en fonction de la demande du marché du travail, de la disponibilité de compétences spécifiques dans certaines régions et d'autres facteurs socio-économiques.")

st.text("")
st.text("")
st.text("")

columns_pr = ['Nom de la Région', 'Nombre de projet de recrutement', 'Nombre de projet de recrutement jugés difficile', 'Nombre de projet de recrutement saisonniers']
columns_pr_selected = new_data[columns_pr]

def to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce').astype(int)

to_numeric(columns_pr_selected, 'Nombre de projet de recrutement')
to_numeric(columns_pr_selected, 'Nombre de projet de recrutement jugés difficile')
to_numeric(columns_pr_selected, 'Nombre de projet de recrutement saisonniers')

somme_par_region_selected = columns_pr_selected.groupby('Nom de la Région').sum()

st.bar_chart(somme_par_region_selected)

st.text("")
st.text("")
st.text("")

st.markdown("Effectivement, en examinant le top 15 des régions avec le plus de projets de recrutement en pourcentage, on constate que même la région avec le pourcentage le plus élevé ne représente qu'une part relativement faible de l'ensemble des projets de recrutement en cours en France. Avec seulement **:red[16%]** de tous les projets de recrutement, il est clair qu'il existe une répartition géographique diversifiée des opportunités d'emploi à travers le pays.")

st.text("")
st.text("")
st.text("")

pourcentage_par_region = somme_par_region_asc / 2002597 * 100
print(somme_par_region_asc)
top_regions_pourc = pourcentage_par_region.head(30)

st.bar_chart(top_regions_pourc)


st.text("")
st.text("")
st.text("")


st.markdown('<h5>Conclusion</h5>', unsafe_allow_html=True)

st.markdown("Nous pouvons en conlure que l'analyse des projets de recrutement en France met en évidence plusieurs points clés.<br>Premièrement, l'Île-de-France se distingue en tant que région avec le plus grand nombre de projets de recrutement, ce qui témoigne de son importance économique et de son attractivité pour les entreprises.<br>Deuxièmement, on observe une certaine concentration des projets de recrutement dans les régions Auvergne-Rhône-Alpes, Nouvelle-Aquitaine, Occitanie et Provence-Alpes-Côte d'Azur, ce qui peut être lié à leur dynamisme économique et à la présence d'industries spécifiques.<br>Troisièmement, la diversité des projets de recrutement dans différentes régions indique des besoins variés en termes de compétences et de profils recherchés.", unsafe_allow_html=True)

st.text("")
st.text("")
st.text("")



#Les secteurs sous tension

st.markdown('<h4>Les secteurs en tension</h4>', unsafe_allow_html=True)
st.caption('Selectionner les colonnes pertinentes')
code = '''colonnes = ['Nombre de projet de recrutement', 'Nombre de projet de recrutement jugés difficile', 'Nombre de projet de recrutement saisonniers']'''
st.code(code, language='python')

st.text("")
st.text("")
st.text("")

st.markdown('<h5>Heatmap Famille de métier</h5>', unsafe_allow_html=True)
st.markdown('<p>Nous pouvons constater les différentes catégories, dont les métiers sont le plus rechercher</p>', unsafe_allow_html=True)

st.text("")
st.text("")

columns_metier = ['Label Famille Métier','Nombre de projet de recrutement jugés difficile']
columns_metier_selected = new_data[columns_metier]
columns_metier_selected.head(30)

def to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce').astype(int)

to_numeric(columns_metier_selected, 'Nombre de projet de recrutement jugés difficile')

metier_bmo = columns_metier_selected.groupby('Label Famille Métier').sum()

st.set_option('deprecation.showPyplotGlobalUse', False)


plt.figure(figsize=(12, 8))
sns.heatmap(metier_bmo, annot=True, fmt="d", cmap='YlOrRd')

st.pyplot()


plt.xlabel('Label Famille Métier')
plt.ylabel('')

plt.show()

st.text("")
st.text("")
st.text("")


st.markdown('<h5>Secteur sous hautes tensions</h5>', unsafe_allow_html=True)
st.markdown('Plus de **:blue[ 45% ]** du secteur liée à la vente, au tourisme et aux services, se trouve en galére pour recruter du personnel, sur un total de **:red[1 222 443 ]** de cprojet de recrutement étant jugés difficile. Avec un delta de **:green[780 154]** demande de contrat par rapport au demande de contrat totale France')


nbr_p_diff = new_data['Nombre de projet de recrutement jugés difficile']
nbr_p_diff_int = pd.to_numeric(nbr_p_diff, errors='coerce').astype(int)
nbr_p_diff_sum = nbr_p_diff_int.sum()

metier_bmo_pourc = metier_bmo / nbr_p_diff_sum * 100
metier_bmo_pourc_round = metier_bmo_pourc.apply(lambda x: round(x, 2))

# Créer le diagramme de camembert avec les données sous forme de tableau 1D
labels = metier_bmo_pourc_round.index
values = metier_bmo_pourc_round.values.flatten()

st.text("")
st.text("")

# Afficher le total du nombre de projets de recrutement jugés difficiles
centered_container = st.container()
with centered_container:
    st.metric(label="Recrutement difficile", value=nbr_p_diff_sum, delta="780 154")


st.text("")
st.text("")
st.text("")
st.text("")

# Créer le diagramme de camembert avec Streamlit
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Assurer un cercle parfait
plt.show()
st.pyplot(fig)

st.text("")
st.text("")
st.text("")
st.text("")


st.markdown('<h5>Les métiers</h5>', unsafe_allow_html=True)
st.markdown('Les métiers qui ont le plus de mal à recruter')

columns_bmo_metier = ['Nom Métier BMO','Nombre de projet de recrutement jugés difficile']
columns_bmo_metier_selected = new_data[columns_bmo_metier]

def to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce').astype(int)

to_numeric(columns_bmo_metier_selected, 'Nombre de projet de recrutement jugés difficile')

nom_metier_bmo = columns_bmo_metier_selected.groupby('Nom Métier BMO').sum()

st.set_option('deprecation.showPyplotGlobalUse', False)


# Trier les données par ordre décroissant
nom_metier_bmo_sort = nom_metier_bmo.sort_values(by='Nombre de projet de recrutement jugés difficile', ascending=False)


# Sélectionner uniquement le top 15
top_15_metiers_bmo = nom_metier_bmo_sort.head(15)

# Créer le diagramme à barres
fig, ax = plt.subplots(figsize=(10, 15))
top_15_metiers_bmo.plot(kind='barh', ax=ax)
plt.xlabel('Nombre de projets de recrutement jugés difficiles')
plt.ylabel('Nom Métier BMO')
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)

st.text("")
st.text("")

# Afficher le diagramme à barres avec Streamlit
st.pyplot(fig)


columns_bmo_metiers = ['Nom Métier BMO','Nombre de projet de recrutement','Nombre de projet de recrutement jugés difficile', 'Nombre de projet de recrutement saisonniers']
columns_bmo_metiers_selected = new_data[columns_bmo_metiers]

to_numeric(columns_bmo_metiers_selected, 'Nombre de projet de recrutement')
to_numeric(columns_bmo_metiers_selected, 'Nombre de projet de recrutement jugés difficile')
to_numeric(columns_bmo_metiers_selected, 'Nombre de projet de recrutement saisonniers')

nom_metiers_bmo = columns_bmo_metiers_selected.groupby('Nom Métier BMO').sum()

# Trier les données par ordre décroissant
nom_metiers_bmo_sort = nom_metiers_bmo.sort_values(by='Nombre de projet de recrutement jugés difficile', ascending=False)

top_15_metiers_bmo_pr = nom_metiers_bmo_sort.head(15)

st.text("")
st.text("")

st.markdown('Le nombre de projets globaux présent dans le cadre de recrutement, ceci est le top 15 des métiers qui ont le plus de mal à recruter au vue du nombre de projet qui ljui sont attribué')

st.text("")
st.text("")

st.area_chart(top_15_metiers_bmo_pr, width=20, height=0)


st.text("")
st.text("")
st.text("")

#Création de carte graphique
st.markdown('<h5>La carte</h5>', unsafe_allow_html=True)
st.markdown("Ci dessous une carte graphique des endroits ou l'ont recrute le plus en France")

columns_map = ['Nom de la Région', 'Nombre de projet de recrutement jugés difficile']
columns_map_selected = new_data[columns_map]
columns_map_selected[columns_map_selected['Nom de la Région'].str.contains('Auvergne-Rhône-Alpes')]
shapefile_path = '/Users/nelson/Desktop/BI/data/gadm41_FRA_shp'
france_shapefile = gpd.read_file(shapefile_path)
france_shapefile[france_shapefile['NAME_1'].str.contains('Auvergne-Rhône-Alpes')]
data_geo = columns_map_selected.merge(france_shapefile, left_on='Nom de la Région', right_on='NAME_1')
data_geo['Nombre de projet de recrutement jugés difficile'] = pd.to_numeric(data_geo['Nombre de projet de recrutement jugés difficile'], errors='coerce')
columns_selected_geo = ['Nom de la Région', 'Nombre de projet de recrutement jugés difficile', 'geometry']
data_geo_selected = data_geo[columns_selected_geo]
data_geo_selected_group = data_geo_selected.groupby('Nom de la Région').aggregate({'geometry': 'first', 'Nombre de projet de recrutement jugés difficile': 'sum'})


# Convertir le DataFrame en GeoDataFrame
gdf = gpd.GeoDataFrame(data_geo_selected_group, geometry='geometry')

# Créer une couche de carte avec la colonne "geometry"
layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf.__geo_interface__,
    opacity=0.8,
    stroked=False,
    filled=True,
    extruded=False,
    get_fill_color=[255, 0, 0],  # Couleur de remplissage
    pickable=True
)

# Créer la carte Pydeck
view_state = pdk.ViewState(
    latitude=48.8566,  # Latitude centrale
    longitude=2.3522,  # Longitude centrale
    zoom=6,  # Niveau de zoom
    pitch=0,  # Angle de vue
    bearing=0  # Orientation de la carte
)

# Afficher la carte Pydeck dans Streamlit
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))












