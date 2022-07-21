from matplotlib import transforms
from dataset import CELEBA_Customized

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import torch
from torch.nn.functional import normalize
import torch.nn as nn


import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image


# plot the image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_part_of_dataset(trainloader):
    dataset = []
    labels = []

    for i, (input, label) in enumerate(trainloader):
        dataset.append(input)
        labels.append(label)
        if i == 9 :
            break

    return dataset, labels

def label_window(number_of_classes, label, row_start, row_end, col_start, col_end):
    temp = [0]*number_of_classes
    neighbourhood = 5
    if (col_end > label[1] - neighbourhood) and (col_start < label[1] + neighbourhood) and (row_end > label[0] - neighbourhood) and (row_start < label[0] + neighbourhood):
        temp[0] = 1
    elif (col_end > label[3] - neighbourhood) and (col_start < label[3] + neighbourhood) and (row_end > label[2] - neighbourhood) and (row_start < label[2] + neighbourhood):
        temp[1] = 1
    elif (col_end > label[5] - neighbourhood) and (col_start < label[5] + neighbourhood) and (row_end > label[4] - neighbourhood) and (row_start < label[4] + neighbourhood):
        temp[2] = 1
    elif (col_end > label[7] - neighbourhood) and (col_start < label[7] + neighbourhood) and (row_end > label[6] - neighbourhood) and (row_start < label[6] + neighbourhood):
        temp[3] = 1
    elif (col_end > label[9] - neighbourhood) and (col_start < label[9] + neighbourhood) and (row_end > label[8] - neighbourhood) and (row_start < label[8] + neighbourhood):
        temp[4] = 1
    else:
        temp[5] = 1

    return temp

def make_window(channel, dataset_channel, class_label_matrix, img, label, size, step, number_of_dimensions, number_of_classes):
    windows = img.unfold(1, size, step).unfold(2, size, step).unfold(3, size, step) # unflod(dimension, size, step)
        
    for i in range(windows.shape[2]):
        row_start, row_end = i*step, i*step + size
        for j in range(windows.shape[3]):
            col_start, col_end = j*step, j*step + size

            dataset_channel.append(windows[0,0,i,j,channel].reshape(number_of_dimensions))      
            class_label_matrix.append(label_window(number_of_classes, label[0], row_start, row_end, col_start, col_end))


def make_dataset(channel, dataset_channel, class_label_matrix, dataset, labels, size, step, number_of_dimensions, number_of_classes):
    for i in range(len(dataset)):
        make_window(channel, dataset_channel, class_label_matrix, dataset[i], labels[i], size, step, number_of_dimensions, number_of_classes)


def initialize_centroids(dataset, centroids, inverted_covariances, covariances, gamma, sigma, threshold, number_of_dimensions):
    number_of_centroids = 0
    for data in dataset:
        if number_of_centroids == 0:
            number_of_centroids += 1
            centroids.append(data)
            covariances.append(sigma*torch.eye(number_of_dimensions, dtype=torch.float64))
            inverted_covariances.append(torch.eye(number_of_dimensions, dtype=torch.float64)/sigma)
        else:
            temp = torch.stack(centroids)
            xc = data-temp
            distances = []
            for i in range(len(centroids)):
                distances.append(xc[i].reshape((1, number_of_dimensions))@inverted_covariances[i]@torch.transpose(xc[i].reshape((1, number_of_dimensions)), 0, 1))
            distances = -1*gamma*torch.tensor(distances)
            RM = torch.exp(distances)
            if max(RM) < threshold:
                number_of_centroids += 1
                centroids.append(data)
                covariances.append(sigma*torch.eye(number_of_dimensions, dtype=torch.float64))
                inverted_covariances.append(torch.eye(number_of_dimensions, dtype=torch.float64)/sigma)


def calculate_membership_matrix(dataset ,centroids, inverted_covariances, gamma, number_of_dimensions):
    list_RM = []
    for i in range(len(centroids)):
        xc = dataset-centroids[i]
        temp = ((xc@inverted_covariances[i])*xc)@torch.ones((number_of_dimensions, 1), dtype=torch.float64)
        mem = torch.exp(-1*gamma*temp)
        list_RM.append(mem)
    #normalize the matirx
    RM = torch.cat(list_RM, dim=1)
    miu_ik = normalize(RM, p=1.0, dim = 1)
    return torch.nan_to_num(miu_ik)

def update_centroids(dataset_channel, class_label_matrix, centroids, inverted_covariances, q, gamma, beta, number_of_dimensions):
    number_of_classes = class_label_matrix.shape[1]
    updated_centroid = []
    miu_ik = calculate_membership_matrix(dataset_channel , centroids, inverted_covariances, gamma, number_of_dimensions)

    for i in range(len(centroids)):
        temp = class_label_matrix*miu_ik[:, [i]]
        cnt = torch.sum(temp, 0).reshape((1, number_of_classes)) 

        if torch.sum(cnt) == 0:
            q[i, :] = torch.zeros((1, number_of_classes))
        else:
            q[i, :] = cnt/torch.sum(cnt)

        Ui = torch.sum(q[i]*class_label_matrix, 1).reshape([len(dataset_channel), 1])*miu_ik[:, [i]] # Pik = torch.sum(q[i]*class_label_matrix, 1).reshape([len(dataset), 1])
        # Ui = miu_ik[:, [i]]
        new_c = torch.nan_to_num((1-beta)*centroids[i] + beta*((torch.transpose(Ui, 0, 1)@dataset_channel)/torch.sum(Ui))) # in case that all the elements in Ui become zero
        updated_centroid.append(new_c)

    return updated_centroid

def calculate_inverted_covariance(covariance):
    covariance = np.array(covariance)
    l, mygamma = np.linalg.eig(covariance) # calculate eigenvalues and eigenvectors
    mylambda = np.zeros((l.shape[0], l.shape[0])) # construct a diagnosal matrix with eigenvalues
    # normalize the matirx
    if l.dtype == "complex":
        l = abs(l)
    sum_of_rows = l.sum()
    if sum_of_rows != 0:
        l_norm = l/sum_of_rows
    else:
        l_norm = l
    for i in range(l.shape[0]):
        if l_norm[i] < 0.01:
            mylambda[i][i] = np.float64(0)
        else:
            mylambda[i][i] = np.float64(1/l[i])
    icm = (mygamma)@(mylambda)@(mygamma.T) # invert of the covariance matrix
    icm = np.float64(icm)
    
    return torch.tensor(icm)

def update_covariances(dataset_channel, class_label_matrix, centroids, covariances, inverted_covariances, q, classes, gamma, beta, number_of_dimensions):
    number_of_classes = class_label_matrix.shape[1]
    miu_ik = calculate_membership_matrix(dataset_channel ,centroids, inverted_covariances, gamma, number_of_dimensions)

    for i in range(len(centroids)):
        temp = class_label_matrix*miu_ik[:, [i]]
        cnt = torch.sum(temp, 0).reshape((1, number_of_classes))

        if torch.sum(cnt) == 0:
            q[i, :] = torch.zeros((1, number_of_classes))
        else:
            q[i, :] = cnt/torch.sum(cnt)
        
        Ui = torch.sum(q[i]*class_label_matrix, 1).reshape([len(dataset_channel), 1])*miu_ik[:, [i]]
        # Ui = miu_ik[:, [i]]
        x_clusteri = dataset_channel - centroids[i]
        temp2 = torch.transpose(Ui*x_clusteri, 0, 1)@x_clusteri
        covariances[i] = (1-beta)*covariances[i] + beta*temp2/torch.sum(Ui)
        covariances[i] = torch.nan_to_num(covariances[i]) # in case that all the elements in Ui become zero

        inverted_covariances[i] = calculate_inverted_covariance(covariances[i])


def suppervised_fuzzy_clustering(dataset, labels, iteration, number_of_dimensions, number_of_classes, size, step, gamma, sigma, threshold, beta):
    centroids_channel = []
    icm_channel = []
    entropy_channel = []
    for color_channel in range(3):
        print(f'color channel {color_channel} started.')
        dataset_channel = []
        class_label_matrix = []

        # centroid and cov of each cluster
        centroids = []
        covariances = []
        inverted_covariances = []

        make_dataset(color_channel, dataset_channel, class_label_matrix, dataset, labels, size, step, number_of_dimensions, number_of_classes)
        dataset_channel = torch.stack(dataset_channel)
        dataset_channel = dataset_channel.type(torch.DoubleTensor)
        class_label_matrix = torch.tensor(class_label_matrix, dtype=torch.float64)

        initialize_centroids(dataset_channel, centroids, inverted_covariances, covariances, gamma, sigma, threshold, number_of_dimensions)
        print(len(centroids))
        q = torch.zeros((len(centroids), number_of_classes), dtype=torch.float64) # matrix that contains the probibilities number of clusters x number of classes

        entropy_curve = []
        for j in range(iteration):
            centroids = update_centroids(dataset_channel, class_label_matrix, centroids, inverted_covariances, q, gamma, beta, number_of_dimensions)
            update_covariances(dataset_channel, class_label_matrix, centroids, covariances, inverted_covariances, q, classes, gamma, beta, number_of_dimensions)
        
        # entropy = calculate_entropy(centroids_channel, icm_channel, gamma, dataset, labels, step, size, number_of_dimensions, number_of_classes)
        # entropy_curve.append(entropy)
        # print(f'{i}th Entropy = {entropy}')
        
        centroids_channel.append(centroids)
        icm_channel.append(inverted_covariances)
        entropy_channel.append(entropy_curve)

        print(f'color channel {color_channel} finished.')

    return centroids_channel, icm_channel

# apply filter --------------------------------------------------------------------------------------
def calculate_mahanalobis_distance(icm, m, x, number_of_dimensions):
    # D^2 = (x-m)^T * C^-1 * (x-m), mahanalobis distance formula
    s = x.reshape((number_of_dimensions, 1))-m.reshape((number_of_dimensions, 1))
    return torch.transpose(s, 0, 1)@icm@s

def calculate_membership(icm, vi, x, gamma, number_of_dimensions):
    distance = calculate_mahanalobis_distance(icm, vi, x, number_of_dimensions)
    return torch.exp(-1*gamma*distance)

def surround_pixel(img, size, step):
    img= img[0]
    windows = []
    windows.append(img[0].unfold(0, size, step).unfold(1, size, step)) # unflod(dimension, size, step)
    windows.append(img[1].unfold(0, size, step).unfold(1, size, step)) # unflod(dimension, size, step)
    windows.append(img[2].unfold(0, size, step).unfold(1, size, step)) # unflod(dimension, size, step)
    return windows

def apply_filter(icm_channel, centroids_channel, cluster_no, gamma, number_of_dimensions, img, size, step):
    windows = surround_pixel(img, size, step)
    len_c = [len(centroids_channel[0]), len(centroids_channel[1]), len(centroids_channel[2])]

    result_r = torch.zeros((windows[0].shape[0], windows[0].shape[1]))
    result_g = torch.zeros((windows[0].shape[0], windows[0].shape[1]))
    result_b = torch.zeros((windows[0].shape[0], windows[0].shape[1]))

    for i in range(windows[0].shape[0]):
        for j in range(windows[0].shape[1]):
            if len_c[0] > cluster_no:
                result_r[i][j] = calculate_membership(icm_channel[0][cluster_no], centroids_channel[0][cluster_no], windows[0][i, j], gamma, number_of_dimensions)
            if len_c[1] > cluster_no:
                result_g[i][j] = calculate_membership(icm_channel[1][cluster_no], centroids_channel[1][cluster_no], windows[1][i, j], gamma, number_of_dimensions)
            if len_c[2] > cluster_no:
                result_b[i][j] = calculate_membership(icm_channel[2][cluster_no], centroids_channel[2][cluster_no], windows[2][i, j], gamma, number_of_dimensions)
    
    result = [result_r, result_g, result_b]
    result = torch.stack(result)
    return result

def calculate_output_clusters(icm_channel, centroids_channel, gamma, number_of_dimensions, img, size, step): # in functionie ke to mikhaiiiiiiiiiiiiiiiiiiiiiiiii
    output = []
    
    len_c = [len(centroids_channel[0]), len(centroids_channel[1]), len(centroids_channel[2])]
    out_num = max(len_c)

    for i in range(out_num):
        output.append(apply_filter(icm_channel, centroids_channel, i, gamma, number_of_dimensions, img, size, step))

    return output

def centroid_softmax(centroids_channel):
    softmax = nn.Softmax(dim=1)

    cr = torch.cat(centroids_channel[0])
    cg = torch.cat(centroids_channel[1])
    cb = torch.cat(centroids_channel[2])

    cr = softmax(cr)
    cg = softmax(cg)
    cb = softmax(cb)
    
    return [cr, cg, cb]


def draw_output(icm_channel, centroids_channel, gamma, number_of_dimensions, img, label, size, step):
    output = calculate_output_clusters(icm_channel, centroids_channel, gamma, number_of_dimensions, img, size, step)
    
    # print(' '.join('%5s' % classes[label]))
    imshow(torchvision.utils.make_grid(img))

    for i in range(len(output)):
        print(f'Cluster {i}th:')
        # imshow(torchvision.utils.make_grid(output[i][0]))
        # imshow(torchvision.utils.make_grid(output[i][1]))
        # imshow(torchvision.utils.make_grid(output[i][2]))
        imshow(torchvision.utils.make_grid(output[i]))

# apply filter --------------------------------------------------------------------------------------

if __name__ == '__main__':
    # load data
    trainset = CELEBA_Customized('./dataset/celeba')
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)

    classes = ('lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth', 'none')
    # classes = ('face')

    # dataset keeps our windows, class_label_matrix keeps the label of windows
    dataset, labels = get_part_of_dataset(trainloader)

    dataset_channel = []
    class_label_matrix = []

    # initialize the centroids and the parameters
    number_of_dimensions = 9
    number_of_classes = 6
    size = 3 # 3x3 window size
    step = 2 # one layer of overlap

    gamma = 1
    sigma = 0.5
    threshold = 0.1
    beta = 1

    iteration = 300

    centroids_channel, icm_channel = suppervised_fuzzy_clustering(dataset, labels, iteration, number_of_dimensions, number_of_classes, size, step, gamma, sigma, threshold, beta)

    with open('centroids.pkl', 'wb') as f:
       pickle.dump(centroids_channel, f)
    
    with open('icm.pkl', 'wb') as handle:
        pickle.dump(icm_channel, handle)

    for i in range(len(dataset)):
        filter_img = dataset[i]
        filter_label = labels[i]
        print(f'Image {i}th:')
        draw_output(icm_channel, centroids_channel, gamma, number_of_dimensions, filter_img, filter_label, 3, 1) # with label

    # with open('centroids.pkl', 'rb') as handle:
    #     centroids_channel = pickle.load(handle)
    # with open('icm.pkl', 'rb') as handle:
    #     icm_channel = pickle.load(handle)

    # img = Image.open("test.jpg")
    # convert_tensor = transforms.ToTensor()
    # img = convert_tensor(img)

    # draw_output(icm_channel, centroids_channel, gamma, number_of_dimensions, img.reshape((1, 3, 218, 178)), labels[0], 3, 1) # with label



