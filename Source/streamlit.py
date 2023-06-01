import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
# from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


def loadData():
	df = pd.read_csv("data_car.csv")
	return df

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

def runKMeans(df, num_clusters):
    le = LabelEncoder()
    # df['Manufacturer'] = le.fit_transform(df['Manufacturer'])
    # df['Model'] = le.fit_transform(df['Model'])
    # df['Category'] = le.fit_transform(df['Category'])
    # df['Fuel type'] = le.fit_transform(df['Fuel type'])
    # df['Gear box type'] = le.fit_transform(df['Gear box type'])
    # df['Drive wheels'] = le.fit_transform(df['Drive wheels'])
    # df['Wheel'] = le.fit_transform(df['Wheel'])
    # df['Color'] = le.fit_transform(df['Color'])
    # df['Leather interior'] = df['Leather interior'].astype(int)
    # df['Turbo'] = df['Turbo'].astype(int)
    column = df.columns.values
    for i in column:
        df[i] = le.fit_transform(df[i])

    # Menentukan jumlah cluster

    # Melakukan klusterisasi dengan K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(df)

    # Menambahkan kolom hasil kluster ke dalam dataframe

    # Menampilkan hasil kluster
    
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


