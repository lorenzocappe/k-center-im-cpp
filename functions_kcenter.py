import math
import numpy as np
import scipy as sp
import sys

from sklearn.metrics import pairwise_distances
from scipy import stats


def diameter_approx_vect(points_vector):
    distances = pairwise_distances(points_vector[0, :], points_vector[1:, :])
    return 2 * distances.max()


def distance_set_vect(point, points_set):
    distances = pairwise_distances(points_set, point)
    return np.min(distances)


def radius_set_vect(points_vector, points_center_set):
    distances = pairwise_distances(points_center_set, points_vector)
    min_distances = np.min(distances, axis=0)
    max_distance = np.max(min_distances)
    return max_distance


def kcenter_gon_vect(points_vector, k):
    solution_set = np.zeros((k, points_vector.shape[1]))
    solution_set[0, :] = points_vector[0, :]

    closest_center = np.zeros((points_vector.shape[0], 2))
    closest_center[:, 0:1] = pairwise_distances(points_vector[:, :], solution_set[0:1, :])

    for i in range(1, k):
        distances = pairwise_distances(points_vector[:, :], solution_set[i-1:i, :])
        distances = np.insert(distances, 1, i, axis=1)

        closest_center = np.where(distances[:, 0:1] < closest_center[:, 0:1], distances, closest_center)
        max_point_ind = np.argmax(closest_center[:, 0:1])

        solution_set[i, :] = points_vector[max_point_ind, :]

    return solution_set


def subroutine_hs(points_vector, dim, gss):
    solution_set = np.empty((dim+1, points_vector.shape[1]))
    solution_set[:] = np.nan

    if points_vector.shape[0] == 0:
        return solution_set, 0

    solution_set[0, :] = points_vector[0, :]
    # xxxxxx = radius_set_vect(points_vector, solution_set[0:1, :])

    num_centri = 1
    j = 0
    while j < points_vector.shape[0] and num_centri < dim + 1:
        current_dist = distance_set_vect(points_vector[j:j+1, :], solution_set[0:num_centri, :])
        if current_dist > 2 * gss:
            solution_set[num_centri, :] = points_vector[j, :]
            num_centri += 1
        j += 1

    # xxxxxx2 = radius_set_vect(points_vector, solution_set[0:num_centri, :])
    return solution_set, num_centri


def round1_map_function_vect(points_vector):
    n = points_vector.shape[0]
    p = 1.0 / math.sqrt(n)

    T_1 = points_vector[1 == sp.stats.bernoulli.rvs(p, size=n)]
    return T_1


def round1_reduce_function_vect(T_1, k, epsilon, min_d, max_d):
    X_1_list = []

    n_gss = 1 + math.ceil(math.log(max_d / min_d, 1 + epsilon))
    print('n_gss: ' + str(n_gss))
    for i in range(0, n_gss):
        guess = min_d * ((1 + epsilon) ** i)
        # print(guess)

        temp_set, temp_num_centri = subroutine_hs(T_1, k, guess)
        # print(temp_set)

        if temp_num_centri <= k:
            X_1_list.append({'guess': guess, 'X_1_set': temp_set, 'dim': temp_num_centri})

    return X_1_list


# def round2_map_function():
#    return


def round2_reduce_function_vect(points_vector, X_1_list):
    n = points_vector.shape[0]
    T_2_list = []  # X_1_list.copy()

    for index in range(len(X_1_list)):
        guess = X_1_list[index]['guess']

        distances = pairwise_distances(points_vector[:, :], X_1_list[index]['X_1_set'][0:X_1_list[index]['dim'], :])
        # print(distances)
        min_distances = np.min(distances, axis=1)
        # print(min_distances)

        temp_set = points_vector[min_distances > 2 * guess]
        # print(temp_set[:,:])

        temp_dict = {'guess': guess, 'T_2_set': temp_set, 'X_1_set': X_1_list[index]['X_1_set'], 'dim': X_1_list[index]['dim']}
        T_2_list.append(temp_dict)

    return T_2_list


def round3_reduce_function_vect(T_2_list, k):
    X_final_list = []

    for index in range(len(T_2_list)):
        dim_X_1 = T_2_list[index]['dim']
        dim_T_2 = T_2_list[index]['T_2_set'].shape[0]

        temp_set = None
        temp_num_centri = 0

        if dim_T_2 > 0:
            temp_set, temp_num_centri = subroutine_hs(T_2_list[index]['T_2_set'], k - dim_X_1, T_2_list[index]['guess'])

        if temp_num_centri + dim_X_1 <= k:
            solution_set = T_2_list[index]['X_1_set'].copy()

            if temp_set is not None:
                for element_index in range(temp_set.shape[0]):
                    if not np.isnan(temp_set[element_index, 0]) and not np.any(np.all(temp_set[element_index, :] == solution_set[:dim_X_1, :], axis=1)):
                        solution_set[dim_X_1, :] = temp_set[element_index, :]
                        dim_X_1 += 1

            X_final_list.append({'guess': T_2_list[index]['guess'], 'set': solution_set, 'dim': dim_X_1})

    return X_final_list


def round4_reduce_function_vect(points_vector, X_final_list, k):
    temp_guess = sys.maxsize
    sol_index = -1
    for index in range(len(X_final_list)):
        if temp_guess > X_final_list[index]['guess']:
            temp_guess = X_final_list[index]['guess']
            sol_index = index

    if X_final_list[sol_index]['dim'] < k:
        solution_set = round4_fill_solution(points_vector, X_final_list[sol_index]['set'], X_final_list[sol_index]['dim'], k)
    else:
        solution_set = X_final_list[sol_index]['set']

    final_solution = {'guess': temp_guess, 'set': solution_set[:-1, :]}
    return final_solution


def round4_fill_solution(points_vector, temp_set, dim, k):
    solution_set = temp_set.copy()

    num_centri = dim
    j = 0
    while j < points_vector.shape[0] and num_centri < k:
        current_dist = distance_set_vect(points_vector[j:j+1, :], solution_set[0:num_centri, :])
        if current_dist > 0 and not np.any(np.all(points_vector[j, :] == solution_set[:, :], axis=1)):
            solution_set[num_centri, :] = points_vector[j, :]
            num_centri += 1
        j += 1

    return solution_set


def kcenter_cpp_vect(points_vector, k, T_len):
    n = len(points_vector)
    ell_cpp = math.ceil(n / T_len)
    k_sub = math.ceil(T_len**2 / n)
    print('n: ' + str(n) + ' T_len: ' + str(T_len) + ' l: ' + str(ell_cpp) + ' k_sub: ' + str(k_sub))
    # print('l*k_sub: ' + str(ell_cpp * k_sub))
    # print('k_sub: ' + str(k_sub))

    temp_set = np.empty((k, points_vector.shape[1]))
    len_temp_set = 0
    for i in range(ell_cpp):
        current_set = subroutine_gon_vect(points_vector, k_sub, T_len, i)
        # print(current_set)

        temp_set = np.vstack((temp_set[:len_temp_set, :], current_set[:, :]))
        len_temp_set += k_sub

    print('|T_CPP|: ' + str(temp_set.shape[0]))
    solution_set = subroutine_gon_vect(temp_set, k, temp_set.shape[0], 0)
    return solution_set


def subroutine_gon_vect(points_vector, k, dim_subvector, offset):
    point_vector_index = offset * dim_subvector
    max_point_vector_index = min(((offset+1) * dim_subvector), points_vector.shape[0])
    # print(max_point_vector_index)

    solution_set = np.zeros((k, points_vector.shape[1]))
    solution_set[0, :] = points_vector[point_vector_index, :]

    closest_center = np.zeros((max_point_vector_index - point_vector_index, 2))
    closest_center[:, 0:1] = pairwise_distances(points_vector[point_vector_index:max_point_vector_index, :], solution_set[0:1, :])
    # print(points_vector[0, :])

    for i in range(1, k):
        distances = pairwise_distances(points_vector[point_vector_index:max_point_vector_index, :], solution_set[i - 1:i, :])
        distances = np.insert(distances, 1, i, axis=1)
        # print(distances)

        closest_center = np.where(distances[:, 0:1] < closest_center[:, 0:1], distances, closest_center)
        # print(closest_center)
        max_point_ind = point_vector_index + np.argmax(closest_center[:, 0:1])
        # print(points_vector[max_point_ind, :])

        solution_set[i, :] = points_vector[max_point_ind, :]

    return solution_set


'''
def min_distance(points_vector):  # O(n^2)
    min_dist = sys.maxsize
    for i in range(len(points_vector) - 1):
        for j in range(i + 1, len(points_vector)):
            current_dist = distance(points_vector[i], points_vector[j])
            if min_dist > current_dist:
                min_dist = current_dist

    return min_dist
'''

# np.any(temp_set[element_index, :] == solution_set[:dim_X_1, :])  # sbagliato
# ((temp_set[element_index, :] == solution_set[:dim_X_1, :]).all(axis=1)).any()  # giusto
# np.all(temp_set[element_index, :] == solution_set[:dim_X_1, :], axis=1)
# np.any(np.all(temp_set[element_index, :] == solution_set[:dim_X_1, :], axis=1))  # equivalente

# temp1 = ((points_vector[j, :] == solution_set[:, :]).all(axis=1)).any()
# temp5 = np.any(np.all(points_vector[j, :] == solution_set[:, :], axis=1))
# temp2 = (points_vector[j, :] == solution_set[:, :]).all(axis=1)
# temp4 = np.all(points_vector[j, :] == solution_set[:, :], axis=1)
# temp3 = points_vector[j, :] == solution_set[:, :]
