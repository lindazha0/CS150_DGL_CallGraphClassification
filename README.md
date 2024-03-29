# CS150_DGL_CallGraphClassification

Final project for the course CS150-02: Deep Graph Learning @Tufts University, 2023 Spring.

Workflow:
- Ground truth: Given two call graphs, calculate the *directed graph editing distance*
- Experiments: split dataset, train, and test:
    - for 2 input <G1, G2>, loss = dist(embedding1, embedding2), which is the predicted distance value generated by the GNN model, taking the two graphs as input and producing two   embedding vectors as intermediate values accordingly.
    - train multiple times and fine-tune with various dataset, to obtain an optimal model with an accuracy score
