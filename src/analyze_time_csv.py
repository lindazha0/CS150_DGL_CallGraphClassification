import pandas as pd
import torch, os
import experiments as exp

TIME_CSV_PATH = "times_for_GED_sbatch_10k.csv"
DATASET_SIZE = 1000  # process labels for 1k dataset
GENERATE_LABELS = False

# reconstructe labels as pt files from csv
def main():
    # load the dataset
    data = pd.read_csv(TIME_CSV_PATH)
    # print(data.head())
    print("csv shape: ", data.shape)
    print("columns: ", data.columns)
    print(f"...describing time_sec...")
    print(data["time_sec"].dtype)
    print(data["time_sec"].describe())
    print(f"...describing distance...")
    print(data["distance"].describe())
    mask = data["distance"] == -1
    print(f"Number of graphs with -1 distance: {mask.sum()}/{data.shape[0]}")

    # reconstructe labels from csv
    if not GENERATE_LABELS:
        return
    train_len, test_len = DATASET_SIZE*0.4, DATASET_SIZE*0.1

    # if train labels already existed, skip, otherwise generate
    train_labels_len = min(data.shape[0], train_len)
    train_labels_len = exp.NUM_LABELS if exp.NUM_LABELS > 0 else train_labels_len
    # train_labels_name = str(train_labels_len)+exp.TRAIN_LABELS[3:]
    # if not os.path.exists(os.path.join(exp.DATA_DIR, train_labels_name)):
    #     train_labels = []
    #     for i in range(train_labels_len):
    #         train_labels.append(data["distance"][i])
    #     torch.save(torch.tensor(train_labels, dtype=torch.float32), os.path.join(exp.DATA_DIR, train_labels_name))
    #     print(f"Saved {train_labels_len} train labels to {train_labels_name}")
    # else:
    #     print(f"{train_labels_name} already existed, skip")
    
    # test labels
    if data.shape[0] <= train_len:
        return
    if os.path.exists(os.path.join(exp.DATA_DIR, exp.TEST_LABELS)):
        print(f"{exp.TEST_LABELS} already existed, skip")
        return
    test_labels_len = min(data.shape[0]-train_len, test_len)
    test_labels, test_labels_name = [], str(test_len)+exp.TEST_LABELS[2:]
    for i in range(train_labels_len, train_labels_len+test_labels_len):
        train_labels.append(data["distance"][i])
    torch.save(torch.tensor(test_labels, dtype=torch.float32), os.path.join(exp.DATA_DIR, test_labels_name))
    print(f"Saved {test_labels_len} test labels to {test_labels_name}")

if __name__ == "__main__":
    main()