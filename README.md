# Customer Prioritization System

A machine learning-based system for prioritizing customers based on their purchasing behavior, order frequency, and spending patterns. This project uses K-means clustering combined with adaptive weight learning to rank customers and identify high-value customer segments.

## 📋 Project Overview

This project implements an intelligent customer prioritization framework that analyzes e-commerce customer data to rank customers by their business value. The system uses unsupervised machine learning to segment customers into clusters and applies a weighted scoring algorithm to determine priority rankings.

### Key Features

- **Data Preprocessing**: Handles missing values using statistical imputation (mode for categorical, mean for numeric)
- **Sequential ID Mapping**: Converts large ID values to manageable sequential identifiers
- **Clustering Analysis**: Uses K-means clustering to segment customers into 3 priority groups
- **Adaptive Scoring**: Implements a learning-based priority calculation algorithm
- **Interactive Web Interface**: Streamlit-based application for real-time customer analysis
- **Brand Analysis**: Identifies top brands and customer preferences

## 🏗️ Project Structure

```
CustomerPrioritization/
├── CustomerPrioritization.ipynb    # Jupyter notebook with full analysis
├── customer_priority_app.py        # Streamlit web application
├── README.md                       # This file
└── ecommerceDataset.csv           # Sample e-commerce dataset (required)
```

## 📊 Dataset

The system expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `event_time` | String | Timestamp of the transaction |
| `order_id` | Integer | Unique order identifier |
| `product_id` | Integer | Product identifier |
| `category_id` | Integer | Product category identifier |
| `category_code` | String | Human-readable category code |
| `brand` | String | Product brand name |
| `price` | Float | Transaction price |
| `user_id` | Integer | Customer/user identifier |

### Dataset Statistics
- **Sample Size**: 2,633,521 transaction records
- **Unique Customers**: ~98,263
- **Date Range**: April 2020 onwards
- **Missing Value Coverage**: Handled for category_id (16.4%), brand (19.2%), price (16.4%), and user_id (78.5%)

## 🔧 Technical Stack

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: K-means clustering and standardization
- **streamlit**: Interactive web application framework
- **numpy**: Numerical computations

## 📈 Workflow

### 1. **Data Collection & Exploration**
- Load e-commerce dataset from CSV
- Generate descriptive statistics
- Identify data shape and range

### 2. **Data Preprocessing**
- **Missing Value Handling**: 
  - Categorical features → Filled with mode value
  - Numeric features → Filled with mean value
- **Feature Engineering**:
  - Sequential ID mapping (converts large IDs to 1, 2, 3, ...)
  - Preserves relationships while reducing computational complexity

### 3. **Aggregation**
- Group transactions by customer (`sequential_user_id`)
- Calculate two key metrics per customer:
  - **Total Orders**: Count of unique orders placed
  - **Total Spending**: Sum of prices across all orders

### 4. **Clustering**
- **Standardization**: MinMax scaling to [0, 1] range
- **Algorithm**: K-means with k=3 clusters
- **Initialization**: k-means++ for better convergence

### 5. **Priority Scoring**
Using adaptive weighted formula:
```
Priority_Score = weight[cluster] × (normalized_orders + normalized_spending)
```

Where:
- `normalized_orders = total_orders / cluster_mean_orders`
- `normalized_spending = total_spending / cluster_mean_spending`

### 6. **Weight Optimization**
An iterative feedback loop updates cluster weights:
```
new_weight = old_weight + learning_rate × (current_score - target_score)
```

**Default Parameters:**
- Learning Rate: 0.1
- Iterations: 10
- Cluster Targets: [0.85, 0.9, 0.88]

## 🚀 Usage

### Option 1: Interactive Web Application

```bash
# Install dependencies
pip install streamlit pandas scikit-learn

# Run the Streamlit app
streamlit run customer_priority_app.py
```

The web interface provides:
1. CSV file uploader for custom datasets
2. Automatic data preprocessing
3. Customer spending visualization
4. Top brands analysis
5. Priority ranking table with scores

### Option 2: Jupyter Notebook

```bash
jupyter notebook CustomerPrioritization.ipynb
```

Run cells sequentially to:
- Explore raw data
- Execute preprocessing pipeline
- Generate insights and visualizations

## 📊 Output

The system generates:
1. **Preprocessed Dataset** (`Order and price dataset.csv`):
   - Cleaned transaction data with sequential IDs
   - Customer aggregated metrics (total orders, total spending)

2. **Priority Rankings**:
   - Ranked list of customers by priority score
   - Descending order (highest priority first)
   - Scores range based on cluster weights and metrics

3. **Analytics**:
   - Top 5 brands by customer frequency
   - Cluster distribution and characteristics
   - Customer segment identification

## 🧠 Algorithm Details

### K-Means Clustering
- **Number of Clusters**: 3 (representing priority tiers)
- **Features Used**: total_orders, price_total_spent
- **Optimal k**: Determined via elbow curve analysis
- **Centroids**: Identified as cluster means in scaled feature space

### Cluster Hierarchy
- **Cluster with highest spending centroid**: Identified for targeted marketing
- **Normalized values**: All metrics normalized relative to cluster means
- **Dynamic weights**: Cluster importance adjusts through feedback loop

### Priority Scoring Function
```python
def calculate_priority_score(data, cluster_means, weights):
    for each customer:
        cluster = customer_cluster
        norm_orders = total_orders / cluster_mean_orders
        norm_spent = total_spending / cluster_mean_spending
        score = weight[cluster] * (norm_orders + norm_spent)
```

## 🔄 Weight Update Mechanism

The system includes an adaptive learning mechanism:

```python
def update_weights(data, cluster_means, weights, learning_rate):
    scores = calculate_priority_score(data, cluster_means, weights)
    cluster_scores = average_scores_per_cluster
    
    for each cluster:
        new_weight = weight[cluster] + learning_rate × (cluster_score - target_score)
```

This ensures cluster weights converge toward target performance levels over iterations.

## 💡 Key Insights

1. **Customer Segmentation**: 3 distinct customer tiers based on behavior
2. **Value Distribution**: Uneven spending across customer base
3. **Brand Preferences**: Top brands concentrated in specific categories
4. **Order Frequency**: Variable order patterns across segments

## ⚙️ Configuration Parameters

Adjustable in `customer_priority_app.py`:

```python
k = 3                    # Number of clusters
learning_rate = 0.1      # Weight update step size
num_iterations = 10      # Feedback loop iterations

weights = {0: 0.8, 1: 1.0, 2: 0.9}    # Initial cluster weights
target_values = {0: 0.85, 1: 0.9, 2: 0.88}  # Target cluster scores
```

## 📈 Performance Considerations

- **Dataset Size**: Efficiently handles 2M+ records
- **Scalability**: Linear with data size (O(n))
- **Memory Usage**: Optimized for standard hardware
- **Execution Time**: Processing time ~seconds for standard datasets

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing value errors | Ensure dataset has required columns |
| Low accuracy | Adjust cluster count (k) or weights |
| Memory issues | Process data in batches or use sampling |
| File format errors | Verify CSV encoding (UTF-8) |

## 📚 Dependencies

```txt
pandas>=1.0.0
scikit-learn>=0.24.0
streamlit>=1.0.0
numpy>=1.18.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Use Cases

1. **Customer Retention**: Identify VIP customers for targeted retention programs
2. **Marketing Budget Allocation**: Optimize spending toward high-value segments
3. **Personalization**: Tailor offers based on customer priority tier
4. **Demand Forecasting**: Predict customer lifetime value
5. **Sales Strategy**: Focus sales efforts on priority customers

## 🔮 Future Enhancements

- [ ] Real-time prediction API
- [ ] Multiple clustering algorithms comparison
- [ ] Temporal analysis (customer lifetime trends)
- [ ] RFM (Recency-Frequency-Monetary) integration
- [ ] Advanced visualization dashboard
- [ ] Model persistence and versioning
- [ ] A/B testing framework for weight optimization

## 📝 License

This project is open source and available for educational and commercial use.

## 👤 Author

Created by: Dipsan49

## 📧 Contact & Support

For questions or issues, please create a GitHub issue in the repository.

---

**Last Updated**: August 2026  
**Version**: 1.0.0  
**Status**: Production Ready
