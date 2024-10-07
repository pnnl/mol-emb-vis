This Dashboard uses multiple tools for comparing and analyzing any number of deep learning model embeddings of the same molecules and their molecular properties
<br/>
The analysis shows the global and local organization of the model<br/>
<br/>
This dashboard is for analyzing embedding of molecules for multiple models<br/>
    Please have at least 11 molecules<br/>
    Please have at least 2 models to compare<br/>
<br/>
Create folders: <br/>
    1. Run the first code block in prepare_data<br/>
Or create the folders manually:<br/>
    1. Create a folder named 'data' on the same directorate level as this program (the dashboard.py program)<br/>
    2. Create five folders inside of the 'data' folder named 'embeddings', 'truth', 'features', 'relationships', and 'prepared_data'<br/>
<br/>
Files needed to run the Dashboard: Please place all files in their respective folders all within the 'data' folder which is on the same directorate level as this file (dashboard.py). ***All files should be .csv files.<br/>
<br/>
Model embeddings: Place all embedding files (one for each deep learning model) into the 'embeddings' folder<br/>
    File names: name of the model which generated the embeddings (Make sure these names are how you want them!! You will have to re-prepare the data if you change these names)<br/>
    File orders: Make sure to order these files within the folder in the order you would like your models to be see on the Dashboard! The programs will pull a list of names from these files, and the order of those names will be used in al the generated files as well as in the Dashboard<br/>
    File properties: - file should contain a number of rows (at least 11), each row being an embedding for a molecule<br/>
                     - make sure the file includes a 'SMILES' column with the smiles string of the molecule each embedding represents<br/>
<br/>
Molecular property descriptors/features: Place a single file for other features for the embeddings into the 'features' folder<br/>
    File properties: - file should contain a number of rows (at least 11), each row consisting of features for a specific molecule<br/>
                     - make sure the file includes a 'SMILES' column with the smiles string of the molecule each embedding represents<br/>
<br/>
Prediction truth values/labels (Optional; You may leave folder empty): Place files for the truth values/labels of the deep learning model predictions into the 'truth' folder<br/>
    File names: name of the molecular property represented by the truth values/labels which your models are trained on<br/>
    File properties: - file should contain two columns: a smiles column named 'SMILES' and a column for the truth values of property of the molecule for each smiles string<br/>
                     - make sure the file includes a 'SMILES' column with the smiles string of the molecule for each truth value<br/>
<br/>
Model relationships (Optional; You may leave folder empty): Place a single file for child/parent relationships between models into the 'relationships' folder<br/>
    File properties: - file should contain two columns named 'child' and 'parent', each row should describe one relationship<br/>
                     - each child should only have one parent, but parents can have multiple children<br/>
                     - make sure the name of each model in the dataframe equals their embedding file names<br/>
<br/>
Generated files:<br/>
    - run the 'prepare_data.ipynb' program to generate the .csv files necessary to run the Dashboard<br/>
    - the generated files will automatically by saved into the 'prepared_data' folder, with all the UMAPs in a subfolder<br/>
    - if folders are not yet created, this file will also create the folders for you so you can input your embedding, features, and truth data<br/>

# Disclaimer Notice
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830