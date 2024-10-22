# Dimensionality's Blessing

## Setup

To get started with the Dimensionality's Blessing project, follow these steps:

1. **Clone the Repository**  
  Clone this repository to your local machine using the following command:
  ```bash
  git clone https://github.com/vishalh21/Dimensionality-s-Blessing
  ```
  
  Navigate into the cloned directory:
  ```bash
  cd Dimensionality-s-Blessing
  ```

2. **Install Dependencies**
  Install the required Python packages using:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the Main Notebook
  - The main code is contained in the `Main.ipynb` notebook. This notebook is designed to run the clustering algorithms on the sample image dataset stored in the `Samples/` directory.
  - The first half of the notebook executes all the clustering methods on the images found in the `Samples/` directory.
  - The second half visualizes the clustering results and provides insights into the clustering performance.
  - Additionally, it processes other datasets by specifying their respective directories for clustering.
  - Clustering results for different methods will be saved in respective folders as `{method}_clustering_results` for the `Samples/`.

4. Other details
   - `cluster.py` contains the distribution based clustering algorithm and `clustering_algos.py` contains other clustering alsorithms for comparision.
   - `extract_featues.py` is used for extracting features from the images using `ResNetFeatureExtractor`.
   - `evaluate.py` is used evaluation of the clustering algorithms using different methods and comparing them.
   - `save_clusters.py` is used for saving the clusters locally for the sample datset.
   - Datatset used here provided in `https://www.kind-of-works.com/source_code.html` along with samples from mnist and caltech dataset.

