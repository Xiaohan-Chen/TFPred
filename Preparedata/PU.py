import numpy as np
from scipy.io import loadmat

RDBdata = ['K004','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working conditions

def _normalization(data, normalization):
    if normalization == "0-1":
        data = (data - data.min()) / (data.max() - data.min())
    return data

def _transformation(sub_data, backbone):              
    if backbone in ("ResNet1D"):
        sub_data = sub_data[np.newaxis, :]
    else:
        raise NotImplementedError("Model {backbone} is not implemented.")
    return sub_data

def read_file(path, filename):
    data = loadmat(path)[filename][0][0][2][0][6][2]
    return data.reshape(-1,)

def PU(datadir, load, data_length, labels, window, normalization, backbone, number):
    path = datadir + "/PU/"
    state = WC[load]
    dataset = {label: [] for label in labels}
    for label in labels:
        filename = state + '_' + RDBdata[label] + '_' + '1'
        subset_path = path + RDBdata[label] + '/' + filename + '.mat'
        mat_data = read_file(subset_path, filename)

        mat_data = _normalization(mat_data, normalization)

        start, end = 0, data_length
        # set the endpoint of data sequence
        length = mat_data.shape[0]
        endpoint = data_length + number * window
        if endpoint > length:
            raise Exception("Sample number {} exceeds signal length.".format(number))

        # split the data and transformation
        while end < endpoint:
            sub_data = mat_data[start : end].reshape(-1,)

            sub_data = _transformation(sub_data, backbone)

            dataset[label].append(sub_data)
            start += window
            end += window
        
        dataset[label] = np.array(dataset[label], dtype="float32")

    return dataset

def PUloader(args):
    label_set_list = list(int(i) for i in args.labels.split(","))
    num_data = args.num_train + args.num_validation + args.num_test
    dataset = PU(args.datadir, args.load, args.data_length, label_set_list, \
        args.window, args.normalization, args.backbone, num_data)

    return dataset