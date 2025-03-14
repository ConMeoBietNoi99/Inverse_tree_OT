# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:22:34 2025

@author: DANG DUY
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#====== Đoạn khởi tạo ========


#w_un=np.zeros(len(w))
w_un=[1,1,3,2,1,3,4,5]
w=[3,2,2,6,5,6,8,10]
w_over=[4,3,3,8,7,7,9,27]
#w_over=np.zeros(len(w))

# Tạo cây
tree = nx.Graph()

# Thêm đỉnh và cạnh
edges = [(1, 2), (1, 3), (1, 4), (3, 5), (3, 6), (4, 7), (4, 8)]
tree.add_edges_from(edges)

# Gắn trọng số cho các cạnh 
edge_weights = {
    (1, 2): 1, (1, 3): 1, (1, 4): 1,
    (3, 5): 1, (3, 6): 1, (4, 7): 1, (4, 8): 1
}


# Cập nhật trọng số của các cạnh
for (u, v), weight in edge_weights.items():
    tree[u][v]['weight'] = weight
#===========================================


#=========Hàm vẽ cây=======================
# Hàm vẽ cây ban đầu lúc chưa có chơi gốc
def draw_initial_tree(tree):
    # Tạo vị trí cho các đỉnh
    pos = nx.spring_layout(tree, seed=42)  # Định vị trí các đỉnh

    # Vẽ đồ thị cây vô hướng
    plt.figure(figsize=(10, 6))
    nx.draw(tree, pos, with_labels=True, node_size=800, 
            node_color="skyblue", font_size=12, font_weight="bold", edge_color="gray", style='solid')
    
    # # Gắn trọng số cung (supply) và cầu (demand) cho mỗi đỉnh
    for node in tree.nodes():
        w_d,w_bt, w_t = w_un[node-1], w[node-1],w_over[node-1]
        plt.text(pos[node][0]+0.1, pos[node][1]-0.08 , f"w_under: {w_d}\n w_bt: {w_bt} \n w_upper: {w_t}",fontsize=10, ha='center', color='red', fontweight='bold')
    
    # Gắn trọng số cho các cạnh
    for u, v, data in tree.edges(data=True):
        weight = data.get('weight', None)
        if weight is not None:
            midpoint = [(pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2]
            plt.text(midpoint[0], midpoint[1], f"l: {weight}", fontsize=12, ha='center', color='blue')

    plt.title("Cái cây QQ", fontsize=16)
    plt.show()

#=============================================



def solve_knapsack(x, sourc, w_s, deman):
    costs=np.array([get_distance(tree,x,y) for y in sourc])
#    print(costs)
    capacity=deman
    bounds=w_s
    B=deman
    n=len(bounds)
    solutions=np.zeros(n)
    for i in np.argsort(costs):
        bi=bounds[i]
        if bi<B:
            solutions[i]=bi
            B-=bi
        else:
            solutions[i]=B
            break
    return np.sum(solutions*costs), solutions


#=====Cây con=====
# Hàm tìm các cây con sau khi bỏ đỉnh gốc
def get_subtrees(tree, root):
    # Bỏ đỉnh gốc khỏi đồ thị
    tree_without_root = tree.copy()
    tree_without_root.remove_node(root)
    
    # Tìm các thành phần liên thông còn lại
    subtrees = list(nx.connected_components(tree_without_root))
    
    return subtrees

# Hàm trả về số lượng cây con và các đỉnh thuộc nó
def get_subtree_info(tree, root):
    subtrees = get_subtrees(tree, root)
    subtree_info = [(len(subtree), list(subtree)) for subtree in subtrees]
    return subtree_info
#==========================


# Hàm trả về khoảng cách từ đỉnh start đến đỉnh end
def get_distance(graph, start, end):
    try:
        distance = nx.dijkstra_path_length(graph, source=start, target=end, weight='weight')
        return distance
    except nx.NetworkXNoPath:
        return float('inf')  # Trả về vô cùng nếu không có đường đi


goc=3
draw_initial_tree(tree)
# In ra số lượng cây con và các đỉnh thuộc cây con đó
subtree_info = get_subtree_info(tree, goc)
print(tree.nodes)
T=[]
L=[]
R=[]
W=sum(w)/2
tempt=0
for i, (size, subtree) in enumerate(subtree_info):
    for j in subtree:
        tempt+=w[j-1]
    if tempt>W:
        R=subtree
    tempt = 0
if len(R)==0:
    print("Điểm chỉ định đã là 1-median")
else:
#    print(R)
    L=[x for x in tree.nodes if x not in R]
#    print(L)
    wL=sum(w[i-1] for i in L)
    wR=sum(w[i-1] for i in R)
    w_L=[]
    w_R=[]
    w_tempt=np.zeros(len(w))
    x=np.zeros(len(w))
    for i in range(len(w)):
        if i+1 in R:
            w_tempt[i]=w[i]-w_un[i]
        else:
            w_tempt[i]=w_over[i]-w[i]
    Delta=wR-wL
#    print('w_tempt',w_tempt)
    T1=solve_knapsack(goc,tree.nodes,w_tempt,Delta)
    #T2=solve_knapsack(goc,L,w_L,Delta)
    optimal_val=T1[0]
    optimal_plan=T1[1]
    print('Optimal value', optimal_val)
    print('Op_weight', w+T1[1])
    #print(f"Cây con {i+1} (số lượng đỉnh: {size}): {subtree}")
    for i in range(len(w)):
        if i+1 in L:
            optimal_plan[i]=w[i]+optimal_plan[i]
        else:
            optimal_plan[i]=w[i]-optimal_plan[i]
    print(optimal_plan)