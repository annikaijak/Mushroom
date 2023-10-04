# Importing libraries
import streamlit as st # To be able to run streamlit app
import pandas as pd # To manipulate dataframes
import numpy as np # To manipulate data
import seaborn as sns # To viualize data
import matplotlib.pyplot as plt # To visualize data
from sklearn.preprocessing import LabelEncoder # To encode categorical variables
from sklearn.preprocessing import StandardScaler # To standardize data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split # To split up data into a training and a test dataset
from sklearn.naive_bayes import GaussianNB # To build SML
from sklearn.tree import DecisionTreeClassifier # To build SML
from sklearn.ensemble import RandomForestRegressor # To build SML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

# Function to load the dataset
@st.experimental_memo  # Cache the function to enhance performance
def load_data():
    # Define the file path
    file_path = 'https://raw.githubusercontent.com/mheine93/mushroom/main/mushrooms_dataset.csv'
    
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)

    # Dropping two columns
    df = df.drop('veil-type', axis = 1)
    df = df.drop('stalk-root', axis = 1)
    
    # Recoding the variables to contain the full words
    df['class'].replace(['p', 'e'], ['poisonous', 'edible'], inplace=True)
    df['cap-shape'].replace(['b', 'c', 'x', 'f', 'k', 's'], ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'], inplace=True)
    df['cap-surface'].replace(['f', 'g', 'y', 's'], ['fibrous', 'grooves', 'scaly', 'smooth'], inplace=True)
    df['cap-color'].replace(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w','y'], ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'], inplace=True)
    df['bruises'].replace(['t', 'f'], ['bruises', 'no'], inplace=True)
    df['odor'].replace(['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'], inplace=True)
    df['gill-attachment'].replace(['a', 'd', 'f', 'n'], ['attached', 'descending', 'free', 'notched'], inplace=True)
    df['gill-spacing'].replace(['c', 'w', 'd'], ['close', 'crowded', 'distant'], inplace=True)
    df['gill-color'].replace(['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'], inplace=True)
    df['gill-size'].replace(['b', 'n'], ['broad', 'narrow'], inplace=True)
    df['stalk-shape'].replace(['e', 't'], ['enlarging', 'tapering'], inplace=True)
    df['stalk-surface-above-ring'].replace(['f', 'y', 'k', 's'], ['fibrous', 'scaly', 'silky', 'smooth'], inplace=True)
    df['stalk-surface-below-ring'].replace(['f', 'y', 'k', 's'], ['fibrous', 'scaly', 'silky', 'smooth'], inplace=True)
    df['stalk-color-above-ring'].replace(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], inplace=True)
    df['stalk-color-below-ring'].replace(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], inplace=True)
    df['veil-color'].replace(['n', 'o', 'w', 'y'], ['brown', 'orange', 'white', 'yellow'], inplace=True)
    df['ring-number'].replace(['n', 'o', 't'], ['none', 'one', 'two'], inplace=True)
    df['ring-type'].replace(['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'], ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheating', 'zone'], inplace=True)
    df['spore-print-color'].replace(['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'], inplace=True)
    df['population'].replace(['a', 'c', 'n', 's', 'v', 'y'], ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'], inplace=True)
    df['habitat'].replace(['g', 'l', 'm', 'p', 'u', 'w', 'd'], ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'], inplace=True)
    
    return df

# Load the data using the defined function
df = load_data()
for col in [df.columns]:
    df[col] = df[col].astype('category')

# Set the app title and sidebar header
st.title("Is the mushroom edible or poisonous?")
st.sidebar.title("Options")
choose = st.sidebar.selectbox(
    "Select an option", 
    ["Information about the interface", 
     "Visualisations", 
     "Mushroom Classifier"]
)

if choose == "Information about the interface":
    st.markdown("""
                With this interface you get some knowledge about mushrooms, that you didn't know you needed! Explore the options in the sidebar to get some visualisations over mushroom features and get a classifier for which mushrooms are edible and not.
        """)

    with st.expander("📊 **Need some help?**"):
                     st.markdown("""
    The goal of this interface is to help you decide if a mushroom is edible or not. The interface has these options
    - An information buttom
    - A visualisation buttom where you can see which features are related to a mushroom being poisonous or edible.
    - A mushroom classifier where you can input information about a mushroom you've seen and check if you can eat it!
    """
    )
                    

if choose == "Visualisations":
    # Displaying the Attrition Analysis header
    st.header("Visualisations")
    
    st.subheader("Countplot of mushroom classes")
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.countplot(data=df, x="class")
    plt.title('Edible vs poisonous mushrooms')
    st.pyplot(plt)

    
    # Dropdown to select the type of visualization
    visualization_option = st.selectbox(
        "Select Visualization", 
        ["Countplot of mushroom classes",
         "Pieplot of mushroom classes",
         "Countplot of mushroom bruises",
         "Countplot of Cap Color",
         "Countplot of Odor",
         "Countplot of Gill Attachment",
         "Countplot of Gill Color",
         "Countplot of Stalk Surface Above Ring",
         "Countplot of Stalk Surface Below Ring",
         "Countplot of Stalk Color Above Ring",
         "Countplot of Stalk Color Below Ring",
         "Countplot of Ring Type",
         "Countplot of Spore Print Color",
         "Countplot of Population",
         "Countplot of Habitat"]   
    )
    
    # Visualizations based on user selection
    if visualization_option == "Countplot of mushroom classes":
        # Making a countplot on the mushroom classes
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")
        sns.countplot(data=df, x="class")
        plt.title('Edible vs poisonous mushrooms')
        st.pyplot(plt)
    
    elif visualization_option == "Pieplot of mushroom classes":
        plt.plot()
        class_pie = df['class'].value_counts()
        mushroom_size = class_pie.values.tolist()
        mushroom_types = class_pie.axes[0].tolist()
        mushroom_labels = 'Edible', 'Poisonous'
        plt.title('Mushroom Class Type Percentange', fontsize=10)
        patches = plt.pie(mushroom_size, labels=mushroom_labels, autopct='%1.1f%%', startangle=150)
        st.pyplot(plt)
        
    elif visualization_option == "Countplot of mushroom bruises":  
        sns.countplot(data=df, x="bruises", hue="class")
        plt.title('Relationship between class and the mushroom having bruises')
        st.pyplot(plt)

    elif visualization_option == "Countplot of Cap Color":  
        sns.countplot(data=df, x="cap-color", hue="class")
        plt.title('Relationship between class and the mushrooms cap color')
        st.pyplot(plt)

    elif visualization_option == "Countplot of Odor":  
        sns.countplot(data=df, x="odor", hue="class")
        plt.title('Relationship between class and the mushrooms odor')
        st.pyplot(plt)
        
    elif visualization_option == "Countplot of Gill Attachment":  
        sns.countplot(data=df, x="gill-attachment", hue="class")
        plt.title('Relationship between class and the mushrooms gill attachment')
        st.pyplot(plt)

    elif visualization_option == "Countplot of Gill Color":  
        sns.countplot(data=df, x="gill-color", hue="class")
        plt.title('Relationship between class and the mushrooms gill color)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Stalk Surface Above Ring":  
        sns.countplot(data=df, x="stalk-surface-above-ring", hue="class")
        plt.title('Relationship between class and the mushrooms stalk surface above ring)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Stalk Surface Below Ring":  
        sns.countplot(data=df, x="stalk-surface-below-ring", hue="class")
        plt.title('Relationship between class and the mushrooms stalk surface below ring)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Stalk Color Above Ring":  
        sns.countplot(data=df, x="stalk-color-above-ring", hue="class")
        plt.title('Relationship between class and the mushrooms stalk color above ring)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Stalk Color Below Ring":  
        sns.countplot(data=df, x="stalk-color-below-ring", hue="class")
        plt.title('Relationship between class and the mushrooms stalk color below ring)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Ring Type":  
        sns.countplot(data=df, x="ring-type", hue="class")
        plt.title('Relationship between class and the mushrooms ring type
        st.pyplot(plt)
                      
    elif visualization_option == "Countplot of Spore Print Color":  
        sns.countplot(data=df, x="spore-print-color", hue="class")
        plt.title('Relationship between class and the mushrooms spore print color)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Population":  
        sns.countplot(data=df, x="population", hue="class")
        plt.title('Relationship between class and the mushrooms population)
        st.pyplot(plt)

    elif visualization_option == "Countplot of Habitat":  
        sns.countplot(data=df, x="habitat", hue="class")
        plt.title('Relationship between class and the mushrooms habitat)
        st.pyplot(plt)

    
    # Display dataset overview
    st.header("Dataset Overview")
    st.dataframe(df.head())   


if choose == "Mushroom Classifier":

    # We labelencode all the columns
    
    # Defining the Label Encoder
    labelencoder = LabelEncoder()
    
    # Creating a new dataframe called df_label
    df_label = df.copy()
    
    # Label Encoding all the columns in df_label
    for column in df_label.columns:
      df_label[column] = labelencoder.fit_transform(df_label[column])
    
    # Looking at the first rows in df_label
    df_label.head()
    
    # Setting our feature variables X
    X = df_label.drop(['class'], axis = 1) # We set axis equal to 1, so that we drop the column and not the row
    
    # Setting our target variable y
    y = df_label['class']
    
    # We normalize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Make a new dataframe containing only the columns, that have a significant impact on the mushroom being edible or not
    df_selected = df_label[['class', 'bruises', 'cap-color', 'odor','gill-attachment', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-type', 'spore-print-color', 'population', 'habitat']]
    
    # Setting our feature variables X
    X = df_selected.drop(['class'], axis = 1)
    
    # Setting our target variable y
    y = df_selected['class']
    
    # We normalize the dataset
    X_scaled = scaler.fit_transform(X)
    
    # Splitting up the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=21, test_size=0.2)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train,y_train)
    
    # print("Test Accuracy: {}%".format(round(rf.score(X_test,y_test)*100,2)))
    
    # Define the mappings globally
    bruises_mapping = {"bruises": 1, "no_bruises": 0}
    cap_color_mapping = {"red": 0, "blue": 1, "orange": 2, "yellow": 3, "brown": 4, "white": 5}
    odor_mapping = {"almond": 0, "anise": 1, "creosote": 2, "fishy": 3, "foul": 4,
                    "musty": 5, "none": 6, "pungent": 7, "spicy": 8}
    gill_attachment_mapping = {"attached": 0, "descending": 1, "free": 2, "notched": 3}
    gill_color_mapping = {"black": 0, "brown": 1, "buff": 2, "chocolate": 3, "gray": 4,
                          "green": 5, "orange": 6, "pink": 7, "purple": 8, "red": 9,
                          "white": 10, "yellow": 11}
    stalk_surface_above_ring_mapping = {"fibrous": 0, "scaly": 1, "silky": 2, "smooth": 3}
    stalk_surface_below_ring_mapping = {"fibrous": 0, "scaly": 1, "silky": 2, "smooth": 3}
    stalk_color_above_ring_mapping = {"brown": 0, "buff": 1, "cinnamon": 2, "gray": 3,
                                     "orange": 4, "pink": 5, "red": 6, "white": 7, "yellow": 8}
    stalk_color_below_ring_mapping = {"brown": 0, "buff": 1, "cinnamon": 2, "gray": 3,
                                     "orange": 4, "pink": 5, "red": 6, "white": 7, "yellow": 8}
    ring_type_mapping = {"evanescent": 0, "flaring": 1, "large": 2, "none": 3, "pendant": 4}
    spore_print_color_mapping = {"black": 0, "brown": 1, "buff": 2, "chocolate": 3,
                                "green": 4, "orange": 5, "purple": 6, "white": 7, "yellow": 8}
    population_mapping = {"abundant": 0, "clustered": 1, "numerous": 2, "scattered": 3,
                          "several": 4, "solitary": 5}
    habitat_mapping = {"grasses": 0, "leaves": 1, "meadows": 2, "paths": 3,
                       "urban": 4, "waste": 5, "woods": 6}
    
    def mushroom_class(bruises, cap_color, odor, gill_attachment, gill_color, stalk_surface_above_ring,
                       stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, ring_type,
                       spore_print_color, population, habitat):
        # Convert input text to numerical values using the mappings
        bruises = bruises_mapping[bruises]
        cap_color = cap_color_mapping[cap_color]
        odor = odor_mapping[odor]
        gill_attachment = gill_attachment_mapping[gill_attachment]
        gill_color = gill_color_mapping[gill_color]
        stalk_surface_above_ring = stalk_surface_above_ring_mapping[stalk_surface_above_ring]
        stalk_surface_below_ring = stalk_surface_below_ring_mapping[stalk_surface_below_ring]
        stalk_color_above_ring = stalk_color_above_ring_mapping[stalk_color_above_ring]
        stalk_color_below_ring = stalk_color_below_ring_mapping[stalk_color_below_ring]
        ring_type = ring_type_mapping[ring_type]
        spore_print_color = spore_print_color_mapping[spore_print_color]
        population = population_mapping[population]
        habitat = habitat_mapping[habitat]
    
        # Construct DataFrame for prediction
        new_df = pd.DataFrame({
          'bruises':[bruises],
          'cap-color':[cap_color],
          'odor':[odor],
          'gill-attachment':[gill_attachment],
          'gill-color':[gill_color],
          'stalk-surface-above-ring':[stalk_surface_above_ring],
          'stalk-surface-below-ring':[stalk_surface_below_ring],
          'stalk-color-above-ring':[stalk_color_above_ring],
          'stalk-color-below-ring':[stalk_color_below_ring],
          'ring-type':[ring_type],
          'spore-print-color':[spore_print_color],
          'population':[population],
          'habitat':[habitat]
        })
    
        # Transform using the scaler and predict using the trained model
        new_values_num = pd.DataFrame(scaler.transform(new_df), columns = new_df.columns, index=[0])
        prediction = rf.predict(new_values_num)
    
        # Return the prediction result
        return "Edible 🍄" if prediction == 0 else "Poisonous 💀"
    
    st.header("Mushroom classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bruis = st.radio(
            'Does the mushroom have bruises?',
            ('bruises','no_bruises'))
        
        cap = st.radio(
            'What color is the mushroom cap?',
            ("red", "blue", "orange", "yellow", "brown", "white"))
        
        od = st.radio(
            'How does the mushroom smell?',
            ("almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"))
        
        gi_ac = st.radio(
            'How are the gills attached?',
            ("attached", "descending", "free", "notched"))
        
        gi_co = st.radio(
            'What color are the gills?',
            ("black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"))
        
        st_su_ab_ri = st.radio(
            'Stalk surface above ring',
            ("fibrous", "scaly", "silky", "smooth"))
        
        st_su_be_ri = st.radio(
            'Stalk surface below ring',
            ("fibrous", "scaly", "silky", "smooth"))
    
    with col2:
        st_co_ab_ri = st.radio(
            'Stalk color above ring',
            ("brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"))
        
        st_co_be_ri = st.radio(
            'Stalk color below ring',
            ("brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"))
        
        ri_ty = st.radio(
            'Ring type',
            ("evanescent", "flaring", "large", "none", "pendant"))
        
        sp_pr_co = st.radio(
            'Spore print color',
            ("black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"))
        
        po = st.radio(
            'Population',
            ("abundant", "clustered", "numerous", "scattered", "several", "solitary"))
        
        ha = st.radio(
            'Habitat',
            ("grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"))
    
    
        if st.button("Predict"):
            result = mushroom_class(bruis, cap, od, gi_ac, gi_co, st_su_ab_ri,
                           st_su_be_ri, st_co_ab_ri, st_co_be_ri, ri_ty,
                           sp_pr_co, po, ha)
            st.text(result)
