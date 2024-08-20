import streamlit as st
import pandas as pd
#from customer_priority_module import calculate_priority_score, update_weights  # Assuming that we have encapsulated our logic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Define Streamlit app
def main():
    st.title('Customer Priority Ranking')

    # File upload for dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read uploaded file as DataFrame
        df = pd.read_csv(uploaded_file)

        # Perform preprocessing steps here (similar to your existing code)
        string_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        for col in string_columns:
            mode_value = df[col].mode()[0]  # Get the mode value for imputation
            df[col].fillna(mode_value, inplace=True)

        for col in numeric_columns:
            mean_value = df[col].mean()  # Get the mean value for imputation
            df[col].fillna(mean_value, inplace=True)

        # Function to assign sequential IDs
        def assign_sequential_ids(data_column):
            id_mapping = {}
            sequential_id = 1
            sequential_ids = []
            for value in data_column:
                if value not in id_mapping:
                    id_mapping[value] = sequential_id
                    sequential_ids.append(sequential_id)
                    sequential_id += 1
                else:
                    sequential_ids.append(id_mapping[value])
            return sequential_ids
        df['sequential_user_id'] = assign_sequential_ids(df['user_id'])
        df['sequential_product_id'] = assign_sequential_ids(df['product_id'])
        df['sequential_order_id'] = assign_sequential_ids(df['order_id'])
        df['sequential_category_id'] = assign_sequential_ids(df['category_id'])

        df.drop(columns=['user_id', 'product_id', 'order_id', 'category_id'], inplace=True)
        # Grouping orders by customer_id and counting the number of orders for each customer
        total_orders = df.groupby('sequential_user_id')['sequential_order_id'].count().reset_index()

        # Renaming the columns for better readability
        total_orders.columns = ['sequential_user_id', 'total_orders']

        customer_spending = df.groupby('sequential_user_id')['price'].sum().reset_index()
        st.subheader("Customer spending for each user:")
        st.dataframe(customer_spending)

        df = pd.merge(df, customer_spending, on='sequential_user_id', suffixes=('', '_total_spent'))
        df = pd.merge(df, total_orders, on='sequential_user_id', suffixes=('', '_per_user'))
        df = df.drop_duplicates(subset='sequential_user_id', keep='first')
        #max_order_index = df['total_orders'].idxmax()
        #df=df.drop(max_order_index)

        df.to_csv("Order and price dataset.csv")
        orders=pd.read_csv("Order and price dataset.csv")

        brand_counts = orders['brand'].value_counts()

        # Sort the counts in descending order
        sorted_brand_counts = brand_counts.sort_values(ascending=False)

        # Print the top brands
        st.subheader("Top brands used by customers")
        st.dataframe(sorted_brand_counts.head())

        orders.drop(columns=['price'], inplace=True)
        #Since price is no longer in use for analysis. However we use the first dataset i.e. ez.csv to evaluate accuracy

        num_unique_customers = orders['sequential_user_id'].nunique()
        #st.header("Number of unique customers")
        #st.write(num_unique_customers)

        features = ['total_orders', 'price_total_spent']
        data = orders[features]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_data = (scaled_data + 1) / 2

        k = 3  #selecting clusters based on the elbow curve
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        # Add cluster labels to the DataFrame
        data['Cluster'] = clusters

        # Calculating cluster centroids
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)

        # Finding the cluster with the highest average values for total_orders and price_total_spent
        cluster_with_highest_spending = centroids[:, 1].argmax()

        orders['Cluster'] = clusters

        # Calculate cluster means
        cluster_means = data.groupby('Cluster').mean()

        

        def calculate_priority_score(data, cluster_means, weights):
   
            priority_scores = []
            for index, row in data.iterrows():
                cluster = row['Cluster']
                total_orders = row['total_orders']
                price_total_spent = row['price_total_spent']
                cluster_mean_orders = cluster_means.loc[cluster, 'total_orders']
                cluster_mean_spent = cluster_means.loc[cluster, 'price_total_spent']
                weight = weights.get(cluster, 0)  # Get weight assigned to the cluster (default to 0 if not found)
                # Normalize values (assuming higher values are better)
                norm_orders = total_orders / cluster_mean_orders
                norm_spent = price_total_spent / cluster_mean_spent
                # Calculate priority score
                priority_score = weight * (norm_orders + norm_spent)
                priority_scores.append(priority_score)
                # Add priority scores to the DataFrame
            data['priority_score'] = priority_scores
            return pd.Series(priority_scores, index=data.index)
    
        def update_weights(data, cluster_means, weights, learning_rate):
   
            priority_scores = calculate_priority_score(data, cluster_means, weights)
            # Calculating average priority score for each cluster
            cluster_scores = data.groupby('Cluster')['priority_score'].mean()
            # Updating weights based on the difference between cluster scores and the target values
            updated_weights = {}
            for cluster in weights:
                updated_weights[cluster] = weights[cluster] + learning_rate * (cluster_scores[cluster] - target_values[cluster])
            return updated_weights

    # Defining initial weights
        weights = {
            0: 0.8,
            1: 1.0,
            2: 0.9,
        }

        target_values = {
            0: 0.85,
            1: 0.9,
            2: 0.88,
        }

        learning_rate = 0.1
        num_iterations=10

    # Performing iterations of the feedback loop
        for i in range(num_iterations):
            # Updating weights
            weights = update_weights(orders, cluster_means, weights, learning_rate)

    # Calculating final priority scores
        priority_scores = calculate_priority_score(orders, cluster_means, weights)

    # Ranking customers by priority score
        priority_ranking = priority_scores.sort_values(ascending=False)
        st.subheader("Ranking customers based on the priority value")
        st.dataframe(priority_ranking)


# Run Streamlit app
if __name__ == '__main__':
    main()

