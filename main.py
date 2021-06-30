import pandas as pd
import numpy as np
import functions_kcenter
# from sklearn.metrics import pairwise_distances

# dataset = 'HIGGS11M7D.csv'  # 7 var
# dataset = 'covertype_zip/data/covertype_csv.csv'  # 54-55 var
# dataset = 'uber_archive/uber-raw-data-apr14.csv'  # 2-4 var
dataset = 'uber_output_norm.csv'  # 2-4 var
print(dataset)
print()

# data = pd.read_csv(dataset, header=None, delimiter=' ')  # higgs
data = pd.read_csv(dataset, header=0, delimiter=',')  # uber - covertype - uber_norm
# print(data)
print(data.shape)

cols = data.shape[1]
# print(cols)

# S = np.matrix(data.iloc[:int(data.shape[0]/2), :].values)  # half array, 7 var
# S = np.matrix(data.iloc[:, 0:cols - 1].values)  # 54 var
# S = np.matrix(data.iloc[:, 1:cols - 1].values)  # 2 var, uber
S = np.matrix(data.iloc[:, 0:cols].values)  # 2 var, uber_norm
# S = np.matrix([[1, -1], [100, -100], [2, -2], [3, -3], [5, 5],
#               [101, -101], [102, -102], [-100, 100], [-101, 101], [-102, 102]])
print(S)
print(S.shape)

dim_max = functions_kcenter.diameter_approx_vect(S)
dim_min = dim_max / 10 ** 7

for epsilon in [0.5, 0.1, 0.01]:
    print('-' * 100)
    print('-' * 100)
    print('epsilon = ' + str(epsilon))

    for K in [5, 10, 20, 30, 40, 50]:
        print('-'*100)
        print('K = ' + str(K))

        if K > S.shape[0]:
            break

        print('diameter: ' + str(dim_max))
        print('min distance: ' + str(dim_min))
        print()

        # TEMP = functions_kcenter.round1_map_function_vect(S)
        # print('|TEMP|: ' + str(TEMP.shape))
        T_1 = functions_kcenter.round1_map_function_vect(S)  # TEMP)
        print('|T_1|: ' + str(T_1.shape))
        if T_1.shape[0] == 0:
            exit(1)

        X_1_list = functions_kcenter.round1_reduce_function_vect(T_1, K, epsilon, dim_min, dim_max)
        print('|X_1_list|: ' + str(len(X_1_list)))
        # print(X_1_list)

        T_2_list = functions_kcenter.round2_reduce_function_vect(S, X_1_list)
        print('|T_2_list|: ' + str(len(T_2_list)))

        max_T_2_len = 0
        sum_T_2_len = 0
        for item in T_2_list:
            sum_T_2_len += item['T_2_set'].shape[0]
            if max_T_2_len < item['T_2_set'].shape[0]:
                max_T_2_len = item['T_2_set'].shape[0]
        print('MAX{|T_2|}: ' + str(max_T_2_len))
        print('SUM{|T_2|}: ' + str(sum_T_2_len))

        X_final_list = functions_kcenter.round3_reduce_function_vect(T_2_list, K)
        print('|X_final_list|: ' + str(len(X_final_list)))

        print()

        solution_set_im = functions_kcenter.round4_reduce_function_vect(S, X_final_list, K)
        radius_im = functions_kcenter.radius_set_vect(S, solution_set_im['set'])
        print('im solution:')
        print(solution_set_im)
        print('guess: ' + str(solution_set_im['guess']))
        print('radius: ' + str(radius_im))
        print('2*guess: ' + str(2*solution_set_im['guess']))
        print()

        solution_set_cpp = functions_kcenter.kcenter_cpp_vect(S, K, max(max_T_2_len, T_1.shape[0]))
        radius_cpp = functions_kcenter.radius_set_vect(S, solution_set_cpp)
        print('cpp solution:')  # max:')
        print(solution_set_cpp)
        print('radius: ' + str(radius_cpp))
        print()

        print(T_1.shape[0])
        print(max_T_2_len)
        print(sum_T_2_len)
        print(solution_set_im['guess'])
        print(radius_im)
        print(radius_cpp)
        print()

'''
A = np.matrix([[2,-2], [101, -101], [-101, 101], [5, 5]])
B = np.matrix([[1, -1], [100, -100], [2, -2], [3, -3], [5, 5],
              [101, -101], [102, -102], [-100, 100], [-101, 101], [-102, 102]])
distances = pairwise_distances(A, B)
min_distances = np.min(distances, axis=0)
max_distance = np.max(min_distances)
'''

'''
A = np.matrix([[2,2,2], [101,101,101], [-101,-101,-101], [5,5,5]])
B = np.matrix([[1,1,1], [2,2,2], [3,3,3], [100,100,100],  [101,101,101], [102,102,102],
              [-100,-100,-100], [-101,-101,-101], [-102,-102,-102], [5,5,5]])
print(functions_kcenter.radius_set_vect(B,A))
print(functions_kcenter.distance_set_vect(B[:-1,:],A[3,:]))
distances = pairwise_distances(B[:-1,:],A[3,:])
print(distances)
'''

'''
for item in X_final_list:
    print(item['dim'])
    print('2gss: ' + str(2 * item['guess']))
    print('rad: ' + str(functions_kcenter.radius_set_vect(S, item['set'][:item['dim'], :])))
    distances = pairwise_distances(S[:, :], item['set'][:item['dim'], :])
    min_distances = np.min(distances, axis=0)
    print(np.any(min_distances > 2 * item['guess']))
'''
'''
for item in X_1_list:
    print(item['dim'])
    print('2gss: ' + str(2 * item['guess']))
    print('rad: ' + str(functions_kcenter.radius_set_vect(T_1, item['X_1_set'][:item['dim'], :])))
    print('rad_tot: ' + str(functions_kcenter.radius_set_vect(S, item['X_1_set'][:item['dim'], :])))
    distances = pairwise_distances(T_1[:, :], item['X_1_set'][:item['dim'], :])
    min_distances = np.min(distances, axis=0)
    print(np.any(min_distances > 2 * item['guess']))
'''
