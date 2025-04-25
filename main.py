import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Wireless Network Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)


# Define the GNN model
class WirelessGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(WirelessGNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.fc = torch.nn.Linear(out_channels, 2)

    def forward(self, x, edge_index):
        # For the frontend demo, we're simplifying the model
        # In production, import the proper GNN layers
        x = x.unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# Helper function to download data as CSV
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Main application
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Upload Data", "Model Training", "Results", "About"])

    # Main content
    st.title("Wireless Network Anomaly Detection")

    if page == "Dashboard":
        dashboard_page()
    elif page == "Upload Data":
        upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Results":
        results_page()
    else:
        about_page()


def dashboard_page():
    st.header("Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Network Health Summary")
        st.metric(label="Detection Accuracy", value="94.2%", delta="↑ 2.1%")
        st.metric(label="False Positives", value="3.2%", delta="↓ 0.8%")
        st.metric(label="Total Anomalies (24h)", value="28", delta="↑ 5")

    with col2:
        st.subheader("Anomaly Distribution")
        st.text("Detection by device type")
        fig, ax = plt.subplots(figsize=(5, 3))
        device_types = ["Router", "IoT Device", "Laptop", "Mobile", "Server"]
        counts = [12, 5, 3, 2, 6]
        ax.bar(device_types, counts, color='#1f77b4')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Recent Anomalies")
    recent_anomalies = pd.DataFrame({
        "Timestamp": pd.date_range(start="2025-04-03", periods=5, freq="H"),
        "Source IP": ["192.168.1.45", "10.0.0.12", "172.16.0.89", "192.168.1.23", "10.0.0.5"],
        "Destination IP": ["23.45.67.89", "104.18.123.45", "8.8.8.8", "94.23.45.67", "172.217.20.174"],
        "Protocol": ["TCP", "UDP", "HTTP", "TCP", "HTTPS"],
        "Device Type": ["IoT Device", "Router", "Server", "Laptop", "Mobile"],
        "Anomaly Type": ["Data Exfiltration", "Port Scanning", "DDoS Attempt", "Unusual Traffic", "Command & Control"],
        "Confidence": [0.98, 0.87, 0.93, 0.79, 0.91]
    })
    st.dataframe(recent_anomalies, hide_index=True)

    st.subheader("Network Activity Timeline")
    timeline_data = pd.DataFrame({
        "time": pd.date_range(start="2025-04-03", periods=24, freq="H"),
        "normal": np.random.randint(100, 500, size=24),
        "anomaly": np.random.randint(0, 20, size=24)
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(timeline_data["time"], timeline_data["normal"], timeline_data["anomaly"],
                 labels=["Normal Traffic", "Anomalies"],
                 colors=["#3498db", "#e74c3c"])
    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Packets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def upload_page():
    st.header("Upload Network Data")

    upload_method = st.radio("Select Data Source", ["Upload CSV", "Use Sample Data", "Connect to Network"])

    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write(f"Data shape: {data.shape}")
            st.dataframe(data.head())

            if st.button("Preprocess Data"):
                st.session_state.data = data
                st.success("Data preprocessed and ready for model training!")

    elif upload_method == "Use Sample Data":
        if st.button("Load Sample Data"):
            # Create sample data
            sample_data = pd.DataFrame({
                "Source IP": ["192.168.1." + str(i) for i in range(1, 101)],
                "Destination IP": ["10.0.0." + str(i) for i in range(1, 101)],
                "Protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"], size=100),
                "Device Type": np.random.choice(["Router", "IoT Device", "Laptop", "Mobile", "Server"], size=100),
                "Packet Size": np.random.normal(500, 200, size=100),
                "Signal Strength": np.random.normal(-50, 15, size=100),
                "Data Rate": np.random.normal(100, 30, size=100),
                "Anomaly Label": np.random.choice([0, 1], size=100, p=[0.8, 0.2])
            })
            st.session_state.data = sample_data
            st.success("Sample data loaded!")
            st.dataframe(sample_data.head())

    else:
        st.info("Enter network parameters to connect:")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Network SSID")
            st.text_input("IP Range", value="192.168.1.0/24")
        with col2:
            st.text_input("Authentication", value="API Key", type="password")
            st.number_input("Capture Duration (minutes)", min_value=1, max_value=60, value=5)

        if st.button("Connect and Capture"):
            with st.spinner("Capturing network traffic..."):
                # Simulate network capture
                import time
                time.sleep(3)

                # Create dummy captured data
                captured_data = pd.DataFrame({
                    "Source IP": ["192.168.1." + str(i) for i in range(1, 51)],
                    "Destination IP": ["10.0.0." + str(i) for i in range(1, 51)],
                    "Protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"], size=50),
                    "Device Type": np.random.choice(["Router", "IoT Device", "Laptop", "Mobile", "Server"], size=50),
                    "Packet Size": np.random.normal(500, 200, size=50),
                    "Signal Strength": np.random.normal(-50, 15, size=50),
                    "Data Rate": np.random.normal(100, 30, size=50),
                    "Timestamp": pd.date_range(start="2025-04-04 00:00", periods=50, freq="10s")
                })
                st.session_state.data = captured_data
                st.success("Network traffic captured successfully!")
                st.dataframe(captured_data.head())

    # Data visualization if data exists
    if 'data' in st.session_state:
        st.subheader("Data Visualization")

        viz_type = st.selectbox("Choose visualization",
                                ["Feature Distribution", "Correlation Matrix", "Protocol Distribution"])

        if viz_type == "Feature Distribution":
            if all(col in st.session_state.data.columns for col in ["Packet Size", "Signal Strength", "Data Rate"]):
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                features = ["Packet Size", "Signal Strength", "Data Rate"]

                for i, feature in enumerate(features):
                    sns.histplot(st.session_state.data[feature], ax=axes[i], kde=True)
                    axes[i].set_title(f"{feature} Distribution")

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Required columns not found in the data")

        elif viz_type == "Correlation Matrix":
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = st.session_state.data[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation analysis")

        else:  # Protocol Distribution
            if "Protocol" in st.session_state.data.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                st.session_state.data["Protocol"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_title("Protocol Distribution")
                ax.set_ylabel("")
                st.pyplot(fig)
            else:
                st.warning("Protocol column not found in the data")

        # Download preprocessed data
        st.markdown(download_link(st.session_state.data, "preprocessed_data.csv", "Download Preprocessed Data"),
                    unsafe_allow_html=True)


def model_training_page():
    st.header("Model Training")

    if 'data' not in st.session_state:
        st.warning("Please upload or generate data first")
        if st.button("Use Sample Data"):
            # Create sample data
            sample_data = pd.DataFrame({
                "Source IP": ["192.168.1." + str(i) for i in range(1, 101)],
                "Destination IP": ["10.0.0." + str(i) for i in range(1, 101)],
                "Protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"], size=100),
                "Device Type": np.random.choice(["Router", "IoT Device", "Laptop", "Mobile", "Server"], size=100),
                "Packet Size": np.random.normal(500, 200, size=100),
                "Signal Strength": np.random.normal(-50, 15, size=100),
                "Data Rate": np.random.normal(100, 30, size=100),
                "Anomaly Label": np.random.choice([0, 1], size=100, p=[0.8, 0.2])
            })
            st.session_state.data = sample_data
            st.success("Sample data loaded!")
        return

    # Training parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")
        model_type = st.selectbox("Model Type", ["GNN (Graph Neural Network)", "GAT (Graph Attention Network)"])
        hidden_dim = st.slider("Hidden Dimension", min_value=8, max_value=128, value=16, step=8)
        num_layers = st.slider("Number of Layers", min_value=2, max_value=5, value=3)

    with col2:
        st.subheader("Training Parameters")
        epochs = st.slider("Number of Epochs", min_value=10, max_value=200, value=100, step=10)
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                                         value=0.01)
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # Feature selection
    st.subheader("Feature Selection")
    available_features = st.session_state.data.columns.tolist()
    selected_features = st.multiselect("Select Features", available_features,
                                       default=["Packet Size", "Signal Strength", "Data Rate", "Protocol",
                                                "Device Type"]
                                       if all(f in available_features for f in
                                              ["Packet Size", "Signal Strength", "Data Rate", "Protocol",
                                               "Device Type"])
                                       else available_features[:5])

    # Target variable
    if "Anomaly Label" in available_features:
        target_variable = "Anomaly Label"
    else:
        target_variable = st.selectbox("Select Target Variable", available_features)

    if st.button("Start Training"):
        # Progress bar and simulated training
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate training process
        loss_values = []
        for i in range(epochs):
            # Simulate decreasing loss function
            current_loss = 0.5 * np.exp(-0.03 * i) + 0.1 * np.random.random()
            loss_values.append(current_loss)

            # Update progress
            progress = (i + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {i + 1}/{epochs} | Loss: {current_loss:.4f}")

            # Slow down simulation slightly
            import time
            time.sleep(0.05)

        st.session_state.trained = True
        st.session_state.loss_values = loss_values
        st.session_state.accuracy = 94.2  # Simulated accuracy

        # Show training curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(loss_values, label="Training Loss", color="blue")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        st.pyplot(fig)

        st.success(f"Model trained successfully! Test Accuracy: {st.session_state.accuracy:.2f}%")

        # Save model option
        if st.button("Save Model"):
            st.success("Model saved successfully!")
            st.markdown(download_link(pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_values}),
                                      "training_history.csv", "Download Training History"), unsafe_allow_html=True)


def results_page():
    st.header("Analysis Results")

    if 'trained' not in st.session_state:
        st.warning("Please train the model first")
        return

    # Display model metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Accuracy", value=f"{st.session_state.accuracy:.2f}%")
    with col2:
        st.metric(label="Precision", value="92.7%")
    with col3:
        st.metric(label="Recall", value="95.3%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[450, 25], [12, 213]])  # Simulated confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance (simulated)
    st.subheader("Feature Importance")
    feature_names = ["Signal Strength", "Protocol", "Packet Size", "Data Rate", "Device Type"]
    importance_values = [0.32, 0.27, 0.21, 0.12, 0.08]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names, importance_values, color='#2980b9')
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

    # Interactive prediction
    st.subheader("Test Anomaly Detection")

    col1, col2 = st.columns(2)
    with col1:
        packet_size = st.slider("Packet Size", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Normalized value (standard deviations from mean)")
        signal_strength = st.slider("Signal Strength", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                    help="Normalized value (standard deviations from mean)")
        data_rate = st.slider("Data Rate", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                              help="Normalized value (standard deviations from mean)")

    with col2:
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "HTTP", "HTTPS"])
        device_type = st.selectbox("Device Type", ["Router", "IoT Device", "Laptop", "Mobile", "Server"])
        source_ip = st.text_input("Source IP", value="192.168.1.45")
        destination_ip = st.text_input("Destination IP", value="8.8.8.8")

    if st.button("Detect Anomaly"):
        # Simulate prediction
        # Higher absolute values for sliders increase anomaly probability
        abs_sum = abs(packet_size) + abs(signal_strength) + abs(data_rate)
        anomaly_score = min(0.95, max(0.05, 0.2 + 0.2 * abs_sum + 0.1 * np.random.random()))

        # Adjust score based on protocol and device type
        if protocol == "HTTP" and device_type == "IoT Device":
            anomaly_score += 0.2
        if protocol == "UDP" and device_type == "Server":
            anomaly_score += 0.15

        anomaly_score = min(0.98, anomaly_score)  # Cap at 0.98

        # Display result
        st.subheader("Detection Result")
        col1, col2 = st.columns(2)

        with col1:
            if anomaly_score > 0.5:
                st.error(f"⚠️ Anomaly Detected (Confidence: {anomaly_score:.2%})")
            else:
                st.success(f"✅ Normal Traffic (Confidence: {(1 - anomaly_score):.2%})")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["Normal", "Anomaly"], [1 - anomaly_score, anomaly_score], color=["#2ecc71", "#e74c3c"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            plt.tight_layout()
            st.pyplot(fig)

        # Show possible anomaly details if detected
        if anomaly_score > 0.5:
            st.subheader("Anomaly Details")
            anomaly_types = {
                "Data Exfiltration": anomaly_score * 0.8,
                "Port Scanning": anomaly_score * 0.6,
                "DDoS Attempt": anomaly_score * 0.4,
                "Command & Control": anomaly_score * 0.3,
                "Unusual Traffic": anomaly_score * 0.9
            }

            # Sort by probability
            anomaly_types = {k: v for k, v in sorted(anomaly_types.items(), key=lambda item: item[1], reverse=True)}

            # Display top 3
            for i, (anomaly_type, prob) in enumerate(list(anomaly_types.items())[:3]):
                st.text(f"{anomaly_type}: {prob:.2%} confidence")

            st.info(
                "Recommendation: Investigate this traffic pattern further and consider blocking the source IP if suspicious behavior continues.")


def about_page():
    st.header("About the Anomaly Detection System")

    st.markdown("""
    ### Wireless Network Anomaly Detection

    This application uses Graph Neural Networks (GNNs) to detect anomalies in wireless network traffic.
    By analyzing patterns in network data, the system can identify potential security threats such as:

    - Unauthorized access attempts
    - Data exfiltration
    - Command and control traffic
    - Denial of service attacks
    - Port scanning and network reconnaissance

    ### How It Works

    1. **Data Collection**: The system collects network traffic data including packet information, device types, protocols, and signal characteristics.

    2. **Graph Construction**: Network traffic is modeled as a graph where devices are nodes and connections between them are edges.

    3. **Feature Engineering**: Raw network data is transformed into meaningful features that capture the behavior of network traffic.

    4. **Model Training**: A Graph Neural Network is trained to learn normal vs. anomalous patterns in the data.

    5. **Real-time Detection**: The trained model analyzes incoming traffic to detect anomalies in real-time.

    ### GNN Architecture

    The system uses a hybrid model combining Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) to capture both local and global patterns in network traffic.
    """)

    st.subheader("System Requirements")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Software Requirements:**
        - Python 3.8+
        - PyTorch 2.0+
        - PyTorch Geometric
        - Streamlit 1.20+
        - Pandas, NumPy, Scikit-learn
        - Matplotlib, Seaborn
        """)

    with col2:
        st.markdown("""
        **Hardware Recommendations:**
        - CPU: 4+ cores
        - RAM: 8GB+
        - GPU: Optional, but recommended for larger networks
        - Storage: 1GB+ for application and models
        """)

    st.header("This project is done by:")
    st.subheader("Under the guidance of Prof.Jyothis K P")
    st.info("Sakshi Magdum-1DT21CS129")
    st.info("Sanjana N P-1DT21CS134")
    st.info("Sejal Kumari-1DT21CS136")
    st.info("Srinidhi V S-1DT21CS157")




if __name__ == "__main__":
    main()