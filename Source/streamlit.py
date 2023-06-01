import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

@st.cache
def loadData():
	df = pd.read_csv("data_car.csv")
	return df

@st.cache
def preprocessData(df):
    # Drop rows with missing values
    df.dropna(inplace=True)
    df = pd.get_dummies(df)
    
    #standard scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    
    # Menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    

    return df_scaled
@st.cache
def runKMeans(df, num_clusters):
    le = LabelEncoder()
    column = df.columns.values
    for i in column:
        df[i] = le.fit_transform(df[i])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(df)

    
    return df,  kmeans.labels_
    

def main():
    st.title('Car Prediction')
    st.subheader("Menerapkan Kecerdasan Buatan: Menjelajahi Dunia Machine Learning")
    data = loadData()
    # st.write(data.columns.values)
    st.write(data)  # Display the loaded data
    
    preprocessed_data = preprocessData(data)  # Preprocess the data
      # Display the preprocessed data
    
    num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10)  # Slider to select the number of clusters
    hasil, labels = runKMeans(preprocessed_data, num_clusters)# Run K-Means Clustering with the selected number of clusters
    st.subheader("Kmeans Clustering")
    data['Cluster'] = labels
    hasil['Cluster'] = labels
    st.write(data)
    st.subheader("Visualisasi Data Cluster")
    fig = px.scatter(hasil, x="Price", y="Mileage", color="Cluster")
    st.plotly_chart(fig, theme=None, use_container_width=True)
if __name__ == '__main__':
    main()


