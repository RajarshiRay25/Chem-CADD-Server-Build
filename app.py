# Import libraries

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from streamlit_ketcher import st_ketcher
from scipy.stats import mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Draw,Lipinski,Descriptors,AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import Descriptors, rdmolops, AllChem, QED, MACCSkeys, rdMolDescriptors
from rdkit.Chem import SDWriter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
# import io
# import re

# Import CA2 Dataset - Manual Testing

df_ca2 = pd.read_csv("./CA2 Chemicals.csv")
print(df_ca2.head())
df_ca2 = df_ca2.dropna().reset_index().drop(columns=["index","Standard Units"])
print(df_ca2.columns)

# Divide the dataset into active and inactive chemicals based on standard ic values

df_ca2['State'] = df_ca2['Standard Value'].apply(lambda x: 'active' if x < 1000 else 'inactive')

# IC50 --> pIC50 
# Step 1: Remove rows where Standard Value is 0
df_ca2 = df_ca2[df_ca2['Standard Value'] > 0]

df_ca2['pIC50'] = 9 - np.log10(df_ca2['Standard Value'])

print(df_ca2.head())

# Streamlit

st.title("**Chem-CADD** : A data-driven toolbox for hit identification and Drug design")
st.sidebar.header("💻Cheminformatics Workflows💊")

sidebar_render = st.sidebar.radio("Navigate to : ",["Home","Dataset Analysis","Chemical Analysis","ADMET Analysis","Molecular Similarity Analysis","Descriptor Calculation","Scaffold Analysis","Molecule Sketcher and Viewer" , "Ligand Preparation and SDF Download", "QSAR Modelling for proteins", "About Us"])
st.sidebar.image("chemcaddlogo.png", use_column_width=True)

if sidebar_render == "Home":
    st.write("#### 🌟 Welcome to **Chem-CADD** : A data-driven toolbox for hit identification and Drug design.")

    st.write("🌐💻👨‍💻 Here, we introduce **ChemCADD**, our online platform that enables users to submit their chemical activity datasets for thorough end-to-end evaluations. Our approach makes it easier to investigate bioactivity ranges and find potential chemical candidates that could block important disease causing target proteins and biological factors likes genes and transcription factors (TF). Researchers can avoid aberrant protein production and treat illnesses associated with gene or TF dysregulation by examining these datasets and learning important information about the possible therapeutic effects of different drugs. This cutting-edge instrument is intended to aid in the search for new drugs and deepen our comprehension of the biological effects of potential chemicals, ultimately leading to better health outcomes.🧑‍🏫👨‍🏫")

    st.write("#### **Chem-CADD** hereby provides the following features related to cheminformatics analysis which the users can perform on their chemical bioactivity datasets ➡️📃 ")


    # Point-wise Features
    features = [
        "🔍 **Dataset Analysis**: A thorough examination of the uploaded data.",
        "⚗️ **Chemical Analysis**: Analyzing the structural properties and activity profiles of the compounds.",
        "💊 **ADMET Analysis**: Evaluating the Absorption, Distribution, Metabolism, Excretion, and Toxicity of the chemical entities.",
        "🔗 **Molecular Similarity Analysis**: Identifying compounds with similar structures and properties.",
        "📊 **Descriptor Calculation**: Obtaining relevant molecular descriptors for the compounds.",
        "🧬 **Scaffold Analysis**: Evaluating the chemical frameworks of the compounds.",
        "✏️ **Molecule Sketcher and Viewer**: Visualizing molecular structures for better understanding.",
        "📥 **Ligand Preparation and SDF Download**: Preparing ligands and allowing for SDF file downloads for further modeling.",
        "📈 **QSAR Modelling for target proteins**: Understanding the quantitative relationship between chemical structure and biological activity."
    ]

    # Display each feature in the list
    for feature in features:
        st.write(feature)

    # Closing remarks
    st.write("Together, these features enable a robust investigation into the inhibition of specific target proteins and other related biological factors, ultimately contributing to the development of therapies for diseases associated with biochemical dysregulation.")


if sidebar_render == "Dataset Analysis":
    st.header("Upload the chemical activity dataset")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature allows users to upload their chemical activity datasets. It provides insights into the overall physical structure of the biological data through user-friendly visualizations, helping users to gain a clearer understanding of the results. 😃")
    # Add sample data format for user

    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )




    uploaded_file = st.file_uploader("Upload Your CSV File.",type=["csv"])
    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2.dropna().reset_index().drop(columns=["index","Standard Units"])
        df_ca2['State'] = df_ca2['Standard Value'].apply(lambda x: 'active' if x < 1000 else 'inactive')
        df_ca2 = df_ca2[df_ca2['Standard Value'] > 0]
        df_ca2['pIC50'] = 9 - np.log10(df_ca2['Standard Value'])
        df_ca2 = df_ca2[df_ca2['pIC50'] > 0]

        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())
            # Create and display the bar chart for active/inactive compounds
        st.write("### Bioactivity Class Distribution")
        
        # Create the Seaborn countplot for the 'State' column
        plt.figure(figsize=(5.5, 5.5))
        sns.countplot(x='State', data=df_ca2, palette='Set2', edgecolor='black')
        
        # Customize labels and title
        plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.title('Distribution of Active and Inactive Samples', fontsize=16, fontweight='bold')
        
        # Display the plot in Streamlit
        st.pyplot(plt)

            # Now create and display the distribution plot for pIC50 values
        st.write("### pIC50 Value Distribution")
        
        plt.figure(figsize=(6, 6))
        sns.histplot(df_ca2['pIC50'], bins=20, kde=True, color='blue', edgecolor='black')
        
        # Customize labels and title for the pIC50 distribution
        plt.xlabel('pIC50', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.title('pIC50 Distribution of Compounds', fontsize=16, fontweight='bold')
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        

        # RDKit Visualization of SMILES using mols2grid

         # Check if 'Smiles' column exists
        if 'Smiles' in df_ca2.columns and 'Molecule ChEMBL ID' in df_ca2.columns:
            # Selector for number of molecules to visualize
            max_molecules = len(df_ca2)
            num_molecules = st.slider("Select number of molecules to visualize:", min_value=1, max_value=max_molecules, value=min(10, max_molecules))

            st.write("### Molecular Structure Visualization")
            
            # Create a grid of molecular images with names
            mols = [Chem.MolFromSmiles(s) for s in df_ca2['Smiles'] if isinstance(s, str)]
            names = df_ca2['Molecule ChEMBL ID'].tolist()  # Use the appropriate column name

            # Filter to the specified number of molecules
            images_with_names = [(Draw.MolToImage(mol), name) for mol, name in zip(mols, names)][:num_molecules]

            # Create a grid layout
            cols = 4  # Number of columns in the grid
            
            # Create a container to display the grid
            for i in range(0, len(images_with_names), cols):
                # Create a list of images and names for the current row
                row_images = images_with_names[i:i + cols]
                col_elements = st.columns(len(row_images))  # Create the columns for this row
                
                for col, (img, name) in zip(col_elements, row_images):
                    col.image(img, use_column_width=True)
                    col.caption(name)
            st.success(" **GREAT !!** Now you can copy the molecule ID for the next analysis",icon="✅")
        else:
            st.write("No valid 'Smiles' or 'Molecule ChEMBL ID' column found in the dataset.")

    else:
        st.warning("Please Upload Correct File")

if sidebar_render == "Chemical Analysis":
    st.header("Chemical Structure Visualization and Pharmacological Analysis")
    st.write("###### This feature allows users to input the desired ChEMBL ID of a specific molecule. Users can visualize the chemical structure and gain insights into the pharmacological and physiological information, enhancing their understanding of the selected molecule. Moreover using Lipinsky based parameters and QED score analysis, users will be able to determine whether the chemical is drug like or not 😮💊")
    user_input = st.text_input("Enter ChEMBL ID")
    
    if user_input:  # Check if the user has provided an input
        # Get the SMILES for the user input
        smiles_user = df_ca2[df_ca2['Molecule ChEMBL ID'] == user_input]['Smiles']
        
        if smiles_user.empty:
            # If the result is empty, show an error alert
            st.error(f"No SMILES found for ChEMBL ID: {user_input}. Please check your input.")
        else:
            # Display the SMILES string if found
            st.write(f"SMILES Fetched for {user_input} - {smiles_user.values[0]}")
            smiles_out = smiles_user.values[0]
            mol_out = Chem.MolFromSmiles(smiles_out)
            img_out = Draw.MolToImage(mol_out)
            # st.write("Image of the Molecule")
            # st.write(img_out)

            # Display Atomic indexes

            def molecule_atom_index(mol_out_idx):
                for atom in mol_out_idx.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())
                return mol_out_idx
            
            # st.write("Image of Molecule with atom index")
            mol_atom_idx_out = molecule_atom_index(mol_out)
            img_atom_idx_out = Draw.MolToImage(mol_atom_idx_out)
            # st.write(img_atom_idx_out)
            

            # Create columns for side by side display
            col1, col2 = st.columns(2)

            # Display the original molecule image in the first column
            with col1:
                st.image(img_out, caption="Molecule without Atom Indices", use_column_width=True)
            
            # Display the molecule with atom indices in the second column
            with col2:
                st.image(img_atom_idx_out, caption="Molecule with Atom Indices", use_column_width=True)

           # Assuming you already have the `mol_out` (RDKit molecule) from SMILES:
            # Extract the Bemis-Murcko scaffold from the molecule
            mol_out_scaffold = Chem.MolFromSmiles(smiles_out)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol_out_scaffold)

            # Get the SMILES representation of the scaffold
            scaffold_smiles = Chem.MolToSmiles(scaffold,canonical=True,isomericSmiles=False)
            st.write(scaffold_smiles)
            # Highlight scaffold atoms and bonds
            scaffold_atoms = scaffold.GetSubstructMatch(mol_out)
            highlight_atoms = list(scaffold_atoms)
            highlight_bonds = []

            for bond in scaffold.GetBonds():
                highlight_bonds.append(bond.GetIdx())

            # Display the original molecule with highlighted scaffold
            img_with_scaffold = Draw.MolToImage(mol_out, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds, highlightColor=(0.8, 0.8, 0.3))

            # Streamlit columns for side-by-side display of original molecule and highlighted scaffold
            col1, col2 = st.columns(2)

            # Display the original molecule image in the first column
            with col1:
                st.image(img_out, caption="Original Molecule", use_column_width=True)

            # Display the molecule with highlighted scaffold in the second column
            with col2:
                st.image(img_with_scaffold, caption=f"Molecule with Highlighted Scaffold ({scaffold_smiles})", use_column_width=True) 

            # Display the molecular properties - Lipinski
            st.write("---")
            st.markdown(f"## RESULTS FOR CHEMBL ID - {user_input}")
            st.write("---")
            # Calculate Molecular weight
            mol_wt = Descriptors.MolWt(mol_out)
            st.write(f"- **Molecular Weight (MW)** : {mol_wt:.2f}")
            # Calculate Hydrogen Bond Acceptors
            num_h_acceptors = Lipinski.NumHAcceptors(mol_out)
            st.write(f"- **Number of Hydrogen Bond Acceptors** : {num_h_acceptors}")
            # Calculate Hydrogen Bond Donors
            num_h_donors = Lipinski.NumHDonors(mol_out)
            st.write(f"- **Number of Hydrogen Bond Donors** : {num_h_donors}")
            # Calculate LogP
            logp = Descriptors.MolLogP(mol_out)
            st.write(f"- **LogP value of the chemical** : {logp: .2f}")
            # Calculate Rotatable Bonds
            rotatable_bonds = Lipinski.NumRotatableBonds(mol_out)
            st.write(f"- **Number of rotatable bonds in the chemical** : {rotatable_bonds}")
            # Calculate TPSA
            tpsa = Descriptors.TPSA(mol_out)
            st.write(f"- **Topological Polar Surface Area (TPSA)**: {tpsa:.2f}")
            # Calculate Drug Likeness QED
            qed_value = Descriptors.qed(mol_out)
            st.write(f"- **QED (Quantitative Estimate of Drug-likeness)**: {qed_value:.2f}")
            
            lipinski_submit = st.button("Submit to analyse Lipinski Properties")
            if lipinski_submit :
                # Check Lipinski compliance
                st.write("### Lipinski's Rule and QED Compliance:")
                if mol_wt < 500 and logp < 5 and num_h_donors < 5 and num_h_acceptors < 10:
                    if qed_value > 0.5:
                        st.success(f"This molecule complies with Lipinski's Rule of Five and has a QED score of {qed_value:.2f}. It may be drug-like.")
                    else:
                        st.warning(f"This molecule complies with Lipinski's Rule of Five, but the QED score is {qed_value:.2f}, indicating moderate drug-likeness.")
                else:
                    if qed_value > 0.5:
                        st.warning(f"This molecule does not comply with Lipinski's Rule of Five, but it has a QED score of {qed_value:.2f}, indicating some potential for drug-likeness.")
                    else:
                        st.warning(f"This molecule does not comply with Lipinski's Rule of Five and has a QED score of {qed_value:.2f}, indicating it may not be drug-like.")
    else:
        st.warning("Please enter a ChEMBL ID to see the corresponding SMILES.")


if sidebar_render == "ADMET Analysis":
    st.header("Upload your chemical dataset containing bioactivity for ADMET analysis")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature enables users to upload their chemical activity datasets for comprehensive ADMET analysis. Users can examine the pharmacokinetic properties of the chemical candidates in their dataset, facilitating statistical analysis and distribution observation of these properties. This functionality provides insights into the overall distribution of ADMET values, enhancing the understanding of the pharmacological information associated with the dataset. 😍")
    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload Your CSV File.",type=["csv"])
        # Function to calculate molecular properties
    def calculate_properties(smiles_list):
        properties = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_weight = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                num_h_donors = Lipinski.NumHDonors(mol)
                num_h_acceptors = Lipinski.NumHAcceptors(mol)
                properties.append((smiles, mol_weight, logp, num_h_donors, num_h_acceptors))
            else:
                properties.append((smiles, None, None, None, None))  # Handle invalid SMILES
        return properties
    
    # Mann-Whitney U Test function
    def mannwhitney(df, descriptor):
        active = df[df['State'] == 'active'][descriptor]
        inactive = df[df['State'] == 'inactive'][descriptor]
        
        # Compare samples
        stat, p = mannwhitneyu(active, inactive)
        
        # Interpret results
        alpha = 0.05
        if p > alpha:
            interpretation = 'Same distribution (fail to reject H0)'
        else:
            interpretation = 'Different distribution (reject H0)'

        results = pd.DataFrame({
            'Descriptor': [descriptor],
            'Statistics': [stat],
            'p-value': [p],
            'Alpha': [alpha],
            'Interpretation': [interpretation]
        })
        return results

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2.dropna().reset_index().drop(columns=["index","Standard Units"])
        df_ca2['State'] = df_ca2['Standard Value'].apply(lambda x: 'active' if x < 1000 else 'inactive')
        df_ca2 = df_ca2[df_ca2['Standard Value'] > 0]
        df_ca2['pIC50'] = 9 - np.log10(df_ca2['Standard Value'])
        df_ca2 = df_ca2[df_ca2['pIC50'] > 0]

        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())

        # Check if 'SMILES' column exists
        if 'Smiles' in df_ca2.columns:
            # Calculate molecular properties
            smiles_list = df_ca2['Smiles'].tolist()
            properties = calculate_properties(smiles_list)

            # Create a DataFrame for the calculated properties
            properties_df = pd.DataFrame(properties, columns=['Smiles', 'Molecular Weight (MW)', 'LogP', 'NumHDonors', 'NumHAcceptors'])

            # Merge the properties DataFrame with the original dataset
            final_df = df_ca2.merge(properties_df, on='Smiles', how='left')
            final_df = final_df.drop_duplicates().reset_index().drop(columns=["index"],axis=1)            # Display the results with calculated properties
            st.write("### Results with Calculated Properties")
            st.write(final_df)

            # Allow user to download the new dataset
            csv = final_df.to_csv(index=False)
            st.download_button(label="Download Results as CSV", data=csv, file_name='admet_analysis_results.csv', mime='text/csv')

            # Analyze the ADMET Distributions
            st.write("## Which ADMET Property you would like to analyze?")
            user_input_admet = st.selectbox(""
                                            ,("Molecular Weight (MW)","LogP","NumHDonors","NumHAcceptors"))
            # Create a boxplot for the selected property
            plt.figure(figsize=(8, 6))
            sns.boxplot(hue='State', y=user_input_admet, data=final_df)
            plt.title(f'Boxplot of {user_input_admet} by Bioactivity Class', fontsize=14, fontweight='bold')
            plt.xlabel('Bioactivity Class', fontsize=12, fontweight='bold')
            plt.ylabel(user_input_admet, fontsize=12, fontweight='bold')

            # Render the plot in Streamlit
            st.pyplot(plt)

            # Perform Mann-Whitney U Test
            mw_results = mannwhitney(final_df, user_input_admet)

            # Display the results of the Mann-Whitney U Test
            st.write("### Mann-Whitney U Test Results")
            st.write(mw_results)
        else:
            st.error("The uploaded CSV file does not contain a 'SMILES' column. Please check your file.")
    else:
        st.warning("Please Upload Correct File")

if sidebar_render == "Molecular Similarity Analysis":
    st.header("Determination of molecular similarity through Tanimoto similarity index algorithm")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature allows users to upload their chemical activity datasets to assess the morphological similarity among the chemical candidates within the dataset. Utilizing the Tanimoto similarity index algorithm, this method computes the similarity of molecules based on the morphological descriptors provided by RDKit. 🧐")


    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload Your CSV File.", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2[["Molecule ChEMBL ID","Smiles"]].dropna()
        df_ca2 = df_ca2.rename(columns={"Molecule ChEMBL ID": "Name", "Smiles": "SMILES"}).drop_duplicates()
        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())

        # Check if the necessary columns exist
        if 'Name' in df_ca2.columns and 'SMILES' in df_ca2.columns:
            compound_names = df_ca2['Name'].tolist()
            smiles_list = df_ca2['SMILES'].tolist()
            
            # Step 2: Generate RDKit Molecule objects from SMILES
            molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

            # Step 3: Compute Morgan fingerprints (e.g., radius 2 and 2048 bits)
            fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in molecules]

            # Step 4: Compute the Tanimoto similarity matrix
            st.write("### Computing Tanimoto Similarity Matrix...")
            n_compounds = len(fingerprints)
            similarity_matrix = []
            for i in range(n_compounds):
                similarities = [TanimotoSimilarity(fingerprints[i], fingerprints[j]) for j in range(n_compounds)]
                similarity_matrix.append(similarities)

            similarity_matrix = pd.DataFrame(similarity_matrix, index=compound_names, columns=compound_names)
            
            # Display similarity matrix
            st.write("### Tanimoto Similarity Matrix")
            st.write(similarity_matrix)

            # # Step 5: Clustering using Agglomerative Clustering
            # distance_matrix = 1 - similarity_matrix
            # st.sidebar.write("Clustering Options")
            # n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)

            # clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
            # clusters = clustering.fit_predict(distance_matrix)

            # # Add cluster information to the original dataframe
            # df_ca2['Cluster'] = clusters
            # st.write("### Clustered Data")
            # st.write(df_ca2[['Name', 'SMILES', 'Cluster']])

            # Step 6: Plot Heatmap of Tanimoto Similarity Matrix
            st.write("### Tanimoto Similarity Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False)
            st.pyplot(plt.gcf())
        else:
            st.error("The uploaded CSV must contain 'SMILES' and 'Name' columns.")
    else:
        st.warning("Please Upload Correct file")

## DESCRIPTORS CALCULATION

# Function to calculate Morgan fingerprints as features
def calculate_morgan_fingerprint(mol, radius=2, n_bits=1024):
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fingerprint)

# Function to calculate MACCS keys
def calculate_maccs_keys(mol):
    if mol is None:
        return None
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)
    return list(maccs_keys)

# Topological descriptor functions
def wiener_index(mol):
    g = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
    return nx.wiener_index(g)

def zagreb_index(mol):
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    degrees = adj_matrix.sum(axis=0)
    return sum(degrees**2)

def tpsa(mol):
    return Descriptors.TPSA(mol)

def kier_hall_chi_indices(mol):
    return {
        'Kappa1': Descriptors.Kappa1(mol),
        'Kappa2': Descriptors.Kappa2(mol),
        'Kappa3': Descriptors.Kappa3(mol),
    }

def balaban_index(mol):
    return Descriptors.BalabanJ(mol)

# Lipinski's descriptors
def lipinski_descriptors(mol):
    return {
        'MolecularWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumRings': Descriptors.RingCount(mol),
    }

# 3D Descriptors (Molecular Volume and Surface Area)
def molecular_volume(mol):
    return rdMolDescriptors.CalcExactMolWt(mol)  # Placeholder; replace if needed

def surface_area(mol):
    return rdMolDescriptors.CalcExactMolWt(mol)  # Placeholder; replace if needed

# QED Score
def qed_score(mol):
    return QED.qed(mol)

# Additional RDKit Descriptors
def additional_descriptors(mol):
    return {
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'MolMR': Descriptors.MolMR(mol),
        'Ipc': Descriptors.Ipc(mol),
        'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
        'SMR_VSA1': Descriptors.SMR_VSA1(mol),
        'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
        'EState_VSA1': Descriptors.EState_VSA1(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'Chi0': Descriptors.Chi0(mol),
        'Chi1': Descriptors.Chi1(mol),
        'Chi2n': Descriptors.Chi2n(mol),
        'Chi2v': Descriptors.Chi2v(mol),
        'Kappa1': Descriptors.Kappa1(mol),
        'Kappa2': Descriptors.Kappa2(mol),
        'Kappa3': Descriptors.Kappa3(mol),
        'HallKierAlpha': Descriptors.HallKierAlpha(mol),
        'BalabanJ': Descriptors.BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'Ipc': Descriptors.Ipc(mol),
    }

# Main function to calculate all descriptors
def calculate_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {}

    # Calculate topological descriptors
    descriptors.update({
        'WienerIndex': wiener_index(mol),
        'ZagrebIndex': zagreb_index(mol),
        'TPSA': tpsa(mol),
        'BalabanIndex': balaban_index(mol)
    })

    # Add Kier-Hall chi indices
    descriptors.update(kier_hall_chi_indices(mol))

    # Add Morgan fingerprints
    morgan_fingerprint = calculate_morgan_fingerprint(mol)
    if morgan_fingerprint is not None:
        for i, bit in enumerate(morgan_fingerprint):
            descriptors[f'MorganBit_{i}'] = bit

    # Add MACCS keys
    maccs_keys = calculate_maccs_keys(mol)
    if maccs_keys is not None:
        for i, bit in enumerate(maccs_keys):
            descriptors[f'MACCSKey_{i}'] = bit

    # Add Lipinski's descriptors
    descriptors.update(lipinski_descriptors(mol))

    # Add 3D Descriptors
    descriptors.update({
        'MolecularVolume': molecular_volume(mol),
        'SurfaceArea': surface_area(mol)
    })

    # Add QED Score
    descriptors['QED'] = qed_score(mol)

    # Add additional RDKit descriptors
    descriptors.update(additional_descriptors(mol))

    return descriptors



if sidebar_render == "Descriptor Calculation":
    st.header("Upload your chemical dataset containing bioactivity for descriptor calculation")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature enables users to upload their chemical activity datasets to calculate a range of molecular descriptors. These descriptors are feature vectors that capture the physiological and chemical properties of molecules, providing valuable insights into their pharmacological characteristics and overall behavior within the dataset. 🤓")

    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload Your CSV File.", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2.dropna().reset_index().drop(columns=["index","Standard Units"])
        df_ca2['State'] = df_ca2['Standard Value'].apply(lambda x: 'active' if x < 1000 else 'inactive')
        df_ca2 = df_ca2[df_ca2['Standard Value'] > 0]
        df_ca2['pIC50'] = 9 - np.log10(df_ca2['Standard Value'])
        df_ca2 = df_ca2[df_ca2['pIC50'] > 0]
        df_ca2 = df_ca2[["Molecule ChEMBL ID","Smiles","pIC50"]].drop_duplicates()
        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())

        if 'Smiles' in df_ca2.columns and 'pIC50' in df_ca2.columns:
            st.write("Data Preview:")
            st.write(df_ca2.head())

            # Process each row to calculate descriptors
            descriptor_data = []
            for index, row in df_ca2.iterrows():
                name = row['Molecule ChEMBL ID']
                smiles = row['Smiles']
                activity = row['pIC50']
                descriptors = calculate_all_descriptors(smiles)
                if descriptors is not None:
                    descriptors['Smiles'] = smiles
                    descriptors['pIC50'] = activity
                    descriptors['Molecule ChEMBL ID'] = name
                    descriptor_data.append(descriptors)

            # Convert descriptor data to DataFrame
            descriptor_df = pd.DataFrame(descriptor_data)

            # Display the descriptors
            st.write("Descriptor Calculation Completed!")
            st.write(descriptor_df.head())

            # Download option for the descriptor CSV
            csv = descriptor_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Descriptor CSV",
                data=csv,
                file_name='all_descriptors.csv',
                mime='text/csv',
            )
        else:
            st.error("The uploaded CSV must contain 'Smiles' and 'pIC50' columns.")
    else:
        st.warning("Please Upload Correct File")


if sidebar_render == "Scaffold Analysis":
    st.header("Upload your chemical dataset containing bioactivity for molecular scaffold analysis")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature allows users to upload their chemical activity datasets to identify the core scaffolds within the chemical structures. Scaffolds represent the essential frameworks or backbones of molecules that play a critical role in defining their biological activity. By identifying these scaffolds, users can gain insights into how the structural features of the compounds contribute to their pharmacological effects and drug-like properties, helping to understand their potential as therapeutic agents against target proteins. 😶‍🌫️")

    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload Your CSV File.", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2.dropna().reset_index().drop(columns=["index","Standard Units"])
        df_ca2['State'] = df_ca2['Standard Value'].apply(lambda x: 'active' if x < 1000 else 'inactive')
        df_ca2 = df_ca2[df_ca2['Standard Value'] > 0]
        df_ca2['pIC50'] = 9 - np.log10(df_ca2['Standard Value'])
        df_ca2 = df_ca2[df_ca2['pIC50'] > 0]
        df_ca2 = df_ca2[["Molecule ChEMBL ID","Smiles","pIC50"]].drop_duplicates()
        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())

        ## Scaffold Extraction methodology

        st.write("### Extracting possible molecular scaffolds")

        # Convert SMILES strings to RDKit Molecule objects
        df_ca2['Molecule'] = df_ca2['Smiles'].apply(lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None)

        # Drop the columns which failed to convert into rdkit objects

        df_ca2 = df_ca2.dropna(subset=["Molecule"])

        # Extracting the Bemis Murcko scaffold for each rdkit Molecule object

        df_ca2["Scaffold"] = df_ca2['Molecule'].apply(MurckoScaffold.GetScaffoldForMol)

        # Convert the scaffold object back to SMILES annotation for implementation

        df_ca2["Scaffold Smiles"] = df_ca2['Scaffold'].apply(Chem.MolToSmiles)

        # Calculate the counts of scaffolds

        scaffold_counts = df_ca2['Scaffold Smiles'].value_counts()

        # See the top 10 scaffolds

        top_10_scaffolds = scaffold_counts.head(10).index


        # Display top 10 scaffolds with counts
        st.write("### Top 10 Scaffolds Based on Frequency")
        st.write(scaffold_counts.head(10))

        # Filter the original dataset to include only those compounds with scaffolds in the top 10
        filtered_df_ca2 = df_ca2[df_ca2['Scaffold Smiles'].isin(top_10_scaffolds)]

        # Display filtered dataset
        st.write("### Compounds with Top 10 Scaffolds")
        st.write(filtered_df_ca2[['Molecule ChEMBL ID', 'Smiles', 'Scaffold Smiles']].head())

        # Visualize the top 10 scaffolds

        # Function to add SMILES and frequency count as text below each molecule image
        def create_labeled_image(mol, smiles, frequency, img_size=(300, 300)):
            # Generate molecule image
            mol_img = Draw.MolToImage(mol, size=img_size)

            # Convert to a PIL image to add text
            img = Image.new('RGB', (img_size[0], img_size[1] + 50), (255, 255, 255))  # Create a larger canvas (for text)
            img.paste(mol_img, (0, 0))  # Paste the molecule image on top
            return img

        # Visualize the top 10 scaffolds with 2 images per row
        st.write("### Top 10 Scaffolds Visualization with SMILES and Frequency")

        # Create a counter to track the number of images displayed
        count = 0

        # Loop through each top scaffold and create individual labeled images
        for i, smiles in enumerate(top_10_scaffolds):
            mol = Chem.MolFromSmiles(smiles)  # Get the molecule
            frequency = scaffold_counts[smiles]  # Get the frequency count

            # Create an image with the molecule, SMILES, and frequency label
            labeled_img = create_labeled_image(mol, smiles, frequency)

        # Create 2 columns layout, display 2 images side by side

    # Create 2 columns layout, display 2 images side by side
            if count % 2 == 0:  # Create new row every two images
                cols = st.columns(2)
            # Display the image and button in the appropriate column
            with cols[count % 2]:
                st.image(labeled_img)

                # Customize the caption with larger font and separate lines for SMILES and frequency
                caption_text = f"""
                <div style='text-align: center;'>
                    <span style='font-size: 20px; font-weight: bold;'>SMILES:</span><br>
                    <span style='font-size: 16px;'>{smiles}</span><br><br>
                    <span style='font-size: 20px; font-weight: bold;'>Frequency:</span><br>
                    <span style='font-size: 16px;'>{frequency}</span>
                </div>
                """
                # Display the caption with markdown and HTML
                st.markdown(caption_text, unsafe_allow_html=True)
                
        # Create three columns for better button alignment (center column will hold the button)
                button_cols = st.columns([1, 1, 1])

                # Place the button in the center column
                with button_cols[1]:
                    # Add the center-aligned working button
                    if st.button(f"View Candidates", key=f"button_{i}"):
                        # Filter the dataset to get all chemicals with the clicked scaffold
                        chemicals_with_scaffold = df_ca2[df_ca2['Scaffold Smiles'] == smiles]
                        st.dataframe(chemicals_with_scaffold[['Molecule ChEMBL ID', 'Smiles', 'pIC50']])
                
            # Increment the counter
            count += 1

        # Optionally, save the filtered dataset to a new CSV file
        st.write("### Download Filtered Dataset")
        csv = filtered_df_ca2.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='filtered_top_10_scaffolds.csv', mime='text/csv')
    else:
        st.warning("Please Upload Correct File")

if sidebar_render == "Molecule Sketcher and Viewer":
    st.header("Molecular Visualiser and Sketcher for Interactive analysis")
    st.write("###### This feature enables users to sketch custom molecules using the inbuilt molecular sketcher. It allows for the intuitive design and visualization of chemical structures, providing flexibility for users to explore various molecular configurations.😍")
    DEFAULT_MOL = "CCC"
    smiles_vis_user = st.text_input("Enter SMILES",DEFAULT_MOL)
    smiles_vis = st_ketcher(smiles_vis_user)
    st.markdown(f"SMILE Code: ``{smiles_vis}``")

if sidebar_render == "Ligand Preparation and SDF Download":

    st.header("Upload your chemical dataset containing SMILES for preparation for Drug Development analysis")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature allows users to upload their chemical datasets and prepare specific chemicals for docking analysis by generating SDF files. The process includes energy minimization, addition of missing molecular components, bond corrections, and other necessary modifications to ensure the molecules are properly optimized. This prepared SDF file is ideal for further computational studies, such as molecular docking and interaction analysis. 🤓")
    sample_data = {
    "Molecule ChEMBL ID": [
        "CHEMBL902", "CHEMBL1566249", "CHEMBL1909049", "CHEMBL309608", 
        "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "Smiles": [
        "NC(N)=Nc1nc(CSCC/C(N)=N/S(N)(=O)=O)cs1",
        "CCOC(=O)OC(C)OC1=C(C(=O)Nc2ccccn2)N(C)S(=O)(=O)c2ccccc21",
        "COc1ccccc1C(=O)N1CCC(Cc2ccccc2)CC1",
        "COCCOC(C)(C)C(=O)Oc1ccc2nc(S(N)(=O)=O)sc2c1",
        "COCC(=O)OCCCS(=O)(=O)c1ccc(S(N)(=O)=O)s1",
        "NS(=O)(=O)NCC1Oc2ccccc2O1",
        "COCCOCCN(CCOC)C(=O)CC1C(=O)Nc2cc(S(N)(=O)=O)sc2S1(=O)=O",
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    ],
    "Standard Type": ["IC50"] * 8,
    "Standard Value": [1.3, 2.3, 50, 1.4, 4.5, 129000, 2.18, 3],
    "Standard Units": ["nM", "nM", "nM", "nM", "nM", "nM", "nM", "nM"]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_chemical_activity_dataset.csv",
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload Your CSV File.", type=["csv"])
    # smiles_input = st.text_input("Enter SMILES string for ligand preparation",placeholder="CCC")
    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df_ca2 = pd.read_csv(uploaded_file)
        df_ca2 = df_ca2.dropna().reset_index().drop_duplicates()
        # Display the first few rows of the dataset
        st.write("### Dataset Preview")
        st.write(df_ca2.head())
    
        # Ligand Preparation Functions
        def smiles_to_3d(smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)  # Add hydrogens
            AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
            AllChem.UFFOptimizeMolecule(mol)  # Optimize the structure with UFF force field
            return mol

        def energy_minimize(mol):
            AllChem.UFFOptimizeMolecule(mol)  # Universal Force Field Minimization
            return mol

        # Streamlit App
        st.title("Ligand Preparation with RDKit")

        # Upload SMILES
        smiles_input = st.text_input("Enter SMILES string for ligand preparation")

        if smiles_input:
            # Step 1: SMILES to 3D Conversion
            mol = smiles_to_3d(smiles_input)
            
            # Step 2: Energy Minimization
            mol = energy_minimize(mol)
            
            
            # Display 3D Molecule
            st.write("### 3D Structure of the Ligand")
            st.image(Draw.MolToImage(mol), caption="Minimized 3D Ligand Structure")
            
            
            # Option to download the prepared ligand as SDF
            if st.button("Write Molecule to SDF"):
                with Chem.SDWriter("prepared_ligand.sdf") as writer:
                    writer.write(mol)
                with open("prepared_ligand.sdf", "rb") as file:
                    st.download_button(
                        label="Download SDF",
                        data=file,
                        file_name="prepared_ligand.sdf",
                        mime="chemical/x-mdl-sdfile"
                    )
    else:
        st.warning("Please Upload Correct File")

if sidebar_render == "QSAR Modelling for proteins":
    st.header("Upload your QSAR dataset for proteins")
    st.warning('Please refer to the sample dataset to understand the required structure for uploading your data.', icon="⚠️")
    st.write("###### This feature enables users to upload their chemical datasets for QSAR (Quantitative Structure-Activity Relationship) analysis using advanced Artificial Intelligence (AI) and Machine Learning (ML) algorithms. Our platform employs predictive modeling techniques that utilize various molecular descriptors, both morphological and functional, to forecast the biological activity—particularly the inhibition potential—of chemical candidates against Carbonic Anhydrase (CA) 2 and 9. The analysis is powered by Random Forest Regression models combined with K-Fold cross-validation to enhance model accuracy and performance. 🚀🔬💡")

    sample_data = {
        "Name": [
        "CHEMBL309608", "CHEMBL320808", "CHEMBL177623", "CHEMBL309950", "CHEMBL18"
    ],
    "nAcid": [0, 0, 0, 0, 0],
    "ALogP": [-0.0256, -1.3806, -1.6122, -2.3511, -0.1473],
    "ALogp2": [6.55E-04, 1.90605636, 2.59918884, 5.52767121, 0.02169729],
    "AMR": [57.3541, 58.0822, 24.7695, 94.8556, 30.5447],
    "pIC50": [10.3, 13, 9.2, 7.4, 6.5]
    }

    # Convert the sample data to a DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Provide download button for the sample file
    st.write("### Download Sample Dataset")
    st.download_button(
        label="Download sample CSV file",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_qsar.csv",
        mime='text/csv'
    )
    uploaded_file_ca = st.file_uploader("Upload Your CSV File.", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file_ca is not None:
        # Load the CSV file into a DataFrame
        df_ca = pd.read_csv(uploaded_file_ca)
        df_ca.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_ca = df_ca.dropna()
        # Display the first few rows of the dataset
        st.write("#### Dataset Preview")
        st.write(df_ca.head())

        # Start processing the dataset
        regression_data = df_ca.copy()
        regression_data = regression_data.drop(['Name'], axis=1)
        regression_data.head()
        st.write("#### Regression QSAR Dataset")
        st.write(regression_data.head())


        # Dataset processing to ensure preservation for pIC50 column during variance threshold

        # Ensure pIC50 is retained by setting aside before filtering
        pIC50 = regression_data["pIC50"]  # Store pIC50 separately
        features_only = regression_data.drop(columns=['pIC50'])  # Drop pIC50 temporarily


        # Variance Threshold
        threshold = VarianceThreshold(threshold=0.1)
        def variance_threshold_selector(data, threshold=0.15):
            selector = VarianceThreshold(threshold)
            selector.fit(data)
            return data[data.columns[selector.get_support(indices=True)]]

        
        # Filter the dataset with only high variance data

        regression_data_HV = variance_threshold_selector(features_only)
        regression_data_HV = pd.concat([regression_data_HV, pIC50], axis=1)

        st.write("#### QSAR Dataset with variance threshold filtering")
        st.write(regression_data_HV.head())

        # Perform correlation analysis now to assess the highly correlated features which we will remove

        correlated_features = set()
        corr_matrix = regression_data_HV.corr()

        # Highly correlated features

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)

        # Remove highly correlated
        def remove_correlated_features(features,data):
            for x in features:
                data.drop(x,axis=1,inplace=True)
            return data

        # Apply removal of highly correlated features
        regression_data_HV_corr = remove_correlated_features(correlated_features,regression_data_HV)
        st.write("#### QSAR Dataset with correlation filtering")
        st.write(regression_data_HV_corr.head())


        # Implement Z score method to obtain samples with proper standard distribution and remove outlier samples

        # Calculate the Z-score for each data point
        z_scores = np.abs(stats.zscore(regression_data_HV_corr['pIC50']))

        # Set a threshold to identify outliers
        threshold = 3  # Standard practice is to remove points with |Z| > 3

        # Filter out the rows where Z-score is greater than the threshold
        regression_data_HV_new = regression_data_HV_corr[(z_scores < threshold)]

        # st.write(f"Original dataset size: {regression_data_HV.shape}")
        # st.write(f"Dataset size after removing outliers: {regression_data_HV_new.shape}")

        # Preparing the data partitions for model training based on features vs target

        X = regression_data_HV_new.drop(['pIC50'], axis=1)
        y = regression_data_HV_new.pIC50

        # Create Train-Test split partition

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize RandomForest Regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=500,          # Number of trees in the forest
            random_state=42,           # Seed for reproducibility
            max_depth=40,              # Maximum depth of each tree
            min_samples_split=5,       # Minimum number of samples required to split an internal node
            min_samples_leaf=2,        # Minimum number of samples required to be at a leaf node
            max_features='sqrt',       # Number of features to consider when looking for the best split
            bootstrap=True,            # Whether bootstrap samples are used when building trees
            oob_score=True,            # Out-of-bag samples to estimate generalization error
            n_jobs=-1,                 # Number of jobs to run in parallel (-1 uses all processors)
            verbose=1                  # Verbose level for monitoring
        )

        # Train the model

        rf_regressor.fit(X_train, y_train)

        # Create Predictions

        if st.button("Make Predictions"):
            # Make predictions on the test data
            y_pred_test = rf_regressor.predict(X_test)

            # Make predictions on the train data
            y_pred_train = rf_regressor.predict(X_train)

            # Calculate regression metrics
            r2_tr = r2_score(y_train, y_pred_train)
            mse_tr = mean_squared_error(y_train, y_pred_train)
            rmse_tr = np.sqrt(mse_tr)
            mae_tr = mean_absolute_error(y_train, y_pred_train)

            # Display metrics
            # st.write(f"#### R² (R-squared) Train Data: {r2_tr}")
            # st.write(f"#### Mean Absolute Error (MAE) Train Data: {mae_tr}")
            # st.write(f"#### Mean Squared Error (MSE) Train Data: {mse_tr}")
            
            r2 = r2_score(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred_test)

            # Display metrics
            # st.write(f"#### R² (R-squared) Test Data: {r2}")
            # st.write(f"#### Mean Absolute Error (MAE) Test Data: {mae}")
            # st.write(f"#### Mean Squared Error (MSE) Test Data: {mse}")

            # Create a DataFrame to display metrics in tabular format
            metrics_data = {
                "Metric": ["R² (R-squared)", "MAE (Mean Absolute Error)", "MSE (Mean Squared Error)", "RMSE (Root Mean Squared Error)"],
                "Train Data": [r2_tr, mae_tr, mse_tr, rmse_tr],
                "Test Data": [r2, mae, mse, rmse]
            }

            # Convert to a DataFrame
            metrics_df = pd.DataFrame(metrics_data)

            # Display the table in Streamlit
            st.write("#### Regression Metrics for Train and Test Data")
            st.table(metrics_df.set_index("Metric"))


            # Create the plot - Train
            plt.figure(figsize=(5,5))
            sns.regplot(x=y_train, y=y_pred_train, line_kws={"color": "red"})
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted Values QSAR - Train Data')
            st.write("#### Regression plot of the QSAR Train data predictions")
            # Display the plot in Streamlit
            st.pyplot(plt)

            # Create the plot - Test
            plt.figure(figsize=(5, 5))
            sns.regplot(x=y_test, y=y_pred_test, line_kws={"color": "red"})
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted Values QSAR - Test Data')
            st.write("#### Regression plot of the QSAR Test data predictions")
            # Display the plot in Streamlit
            st.pyplot(plt)

            # Develop the Applicability Domain

            # Step 1: Standardize the features (mean=0, variance=1)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Step 2: Perform PCA on the training set
            pca = PCA(n_components=2)  # Adjust the number of components to retain relevant variance
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            # Step 3: Calculate Euclidean distance of each test point from the PCA space origin (training mean)
            dist_train = np.linalg.norm(X_train_pca, axis=1)  # Distances of training points in PCA space
            dist_test = np.linalg.norm(X_test_pca, axis=1)    # Distances of test points in PCA space

            # Step 4: Set the Applicability Domain threshold (e.g., 95th percentile of training distances)
            ad_threshold = np.percentile(dist_train, 95)

            # Step 5: Determine whether test points are inside or outside the Applicability Domain
            test_in_ad = dist_test <= ad_threshold  # Boolean mask for points within the AD

            # Step 6: Calculate residuals (errors between actual and predicted values)
            residuals_test = np.abs(y_test - y_pred_test)

            # Step 7: Create Applicability Domain Plot
            plt.figure(figsize=(7, 5))

            # Plot points within the Applicability Domain (green)
            plt.scatter(dist_test[test_in_ad], residuals_test[test_in_ad], color='green', label='Within AD')

            # Plot points outside the Applicability Domain (red)
            plt.scatter(dist_test[~test_in_ad], residuals_test[~test_in_ad], color='red', label='Outside AD')

            # Plot AD threshold line
            plt.axvline(ad_threshold, color='blue', linestyle='--', label=f'AD Threshold: {ad_threshold:.2f}')

            # Labels and title
            plt.xlabel('Distance from Training Set (PCA Space)')
            plt.ylabel('Prediction Residuals (|Actual - Predicted|)')
            plt.title('Applicability Domain Map')
            plt.legend()
            st.write("#### Applicability Domain (AD) of the QSAR Dataset")
            st.pyplot(plt)

    else:
        st.warning("Please Upload Correct File")

if sidebar_render == "About Us":
    st.header("About Us")

    # Team Introduction
    st.write("#### Meet the Team Behind the **Chem-CADD** Server")

    # Team member 1
    st.subheader("👨‍💻 Rajarshi Ray")
    st.write("**Affiliation**: Researcher at Tampere University Finland, Faculty of Medicine and Health Technology")

    # Team member 2
    st.subheader("👨‍🔬 Ratul Bhowmik")
    st.write("**Affiliation**: Researcher at Tampere University Finland, Faculty of Medicine and Health Technology")

    # Team member 3
    st.subheader("👨‍🏫 Dr. Ashok Aspatwar")
    st.write("**Affiliation**: Professor at Tampere University Finland, Faculty of Medicine and Health Technology")

    # Closing statement
    st.write("Together, our team is dedicated to advancing cheminformatics solutions and enabling researchers worldwide to explore chemical bioactivity through cutting-edge computational tools.")